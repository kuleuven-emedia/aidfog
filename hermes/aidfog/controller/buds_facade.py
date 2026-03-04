from abc import ABC, abstractmethod
import asyncio
from collections import deque
from multiprocessing import Queue
from multiprocessing.synchronize import Event as _Event

from bleak import BleakScanner, BleakClient
from bleak.uuids import normalize_uuid_str
from bleak.backends.device import BLEDevice
from bleak.backends.characteristic import BleakGATTCharacteristic

from hermes.utils.time_utils import get_time

from ..utils.types import NiclaData, NiclaPacketMask


class NiclaBackend(ABC):
    @abstractmethod
    async def connect(self):
        pass

    @abstractmethod
    async def run(self):
        pass

    @abstractmethod
    async def cleanup(self):
        pass


class NiclaBleBackend(NiclaBackend):
    def __init__(
        self,
        niclas: dict,
        nicla_latest_data: "dict[str, deque[NiclaData]]",
        nicla_data_queue: "Queue[tuple[str, float, NiclaData]]",
        is_keep_data_event: _Event,
        is_stop_new_data_event: _Event,
        is_cleanup_event: _Event,
    ):
        self._nicla_mac_mapping: dict[str, str] = niclas["device_mapping"]
        self._nicla_latest_data = nicla_latest_data
        self._nicla_data_queue = nicla_data_queue
        self._service_uuid = normalize_uuid_str(niclas["service_uuid"])
        self._char_uuid = normalize_uuid_str(niclas["char_uuid"])
        self._discovered_devices: dict[str, BLEDevice] = {}
        self._connected_devices: dict[str, BleakClient] = {}
        self._disconnected_devices: dict[str, tuple[BleakClient, float]] = {}

        self._is_keep_data_event = is_keep_data_event
        self._is_stop_new_data_event = is_stop_new_data_event
        self._is_cleanup_event = is_cleanup_event

    def _make_data_callback(self, name):
        def callback(
            characteristic: BleakGATTCharacteristic, raw_data: bytearray
        ) -> None:
            toa_s = get_time()
            sample = NiclaData.from_bytes(raw_data)
            self._nicla_latest_data[name].append(sample)

            if (
                self._is_keep_data_event.is_set()
                and not self._is_stop_new_data_event.is_set()
            ):
                self._nicla_data_queue.put((name, toa_s, sample))

        return callback

    def _make_disconnection_callback(self, name: str, device: BLEDevice):
        def callback(client: BleakClient) -> None:
            print(f"Device {name} [{device.address}] disconnected.", flush=True)
            self._connected_devices.pop(name, None)
            self._disconnected_devices[name] = (device, get_time())

        return callback

    async def _discover(self) -> bool:
        discovered_devices = await BleakScanner.discover(
            timeout=10.0, service_uuids=[self._service_uuid]
        )
        found = list(map(lambda device: device.address, discovered_devices))
        if not all([(mac in found) for mac in self._nicla_mac_mapping.values()]):
            not_found = [
                name
                for name, mac in self._nicla_mac_mapping.items()
                if mac not in found
            ]
            print(
                f"Couldn't find {not_found}.\n",
                "Make sure all Niclas are advertising.",
                flush=True,
            )
            return False

        inverted_mac_mapping = {v: k for k, v in self._nicla_mac_mapping.items()}
        self._discovered_devices = {
            inverted_mac_mapping[device.address]: device
            for device in filter(
                lambda d: d.address in self._nicla_mac_mapping.values(),
                discovered_devices,
            )
        }
        return True

    async def _connect_all(self) -> bool:
        try:
            async with asyncio.TaskGroup() as tg:
                client_tasks = [
                    tg.create_task(self._connect_and_subscribe(name, device))
                    for name, device in self._discovered_devices.items()
                ]
            return all([t.result() for t in client_tasks])
        except Exception as e:
            print("Failed to connect to some of the Niclas.\n", e, flush=True)
            return False

    async def _connect_and_subscribe(self, name: str, device: BLEDevice) -> bool:
        client = BleakClient(
            device,
            disconnected_callback=self._make_disconnection_callback(name, device),
        )
        try:
            await client.connect()
            print(f"Connected to {name} [{device.address}]", flush=True)
            await client.start_notify(self._char_uuid, self._make_data_callback(name))
            self._connected_devices[name] = client
            return True
        except Exception as e:
            print(f"Failed to connect to {name}: {e}", flush=True)
            return False

    async def connect(self):
        # Discover IMUs.
        while not (result := await self._discover()):
            print("Trying to rediscover Niclas. Make sure all are on.", flush=True)
            await asyncio.sleep(2)

        # Connect all BLE IMUs.
        while not (result := await self._connect_all()):
            await self.cleanup()
            print("Trying to reconnect to Niclas.", flush=True)
            await asyncio.sleep(2)

    async def run(self):
        while not self._is_cleanup_event.is_set():
            for name, (device, _) in list(self._disconnected_devices.items()):
                print(f"Trying to reconnect to {name}...", flush=True)

                fresh_device = await BleakScanner.find_device_by_address(
                    device.address, timeout=1.0
                )
                if not fresh_device:
                    print(f"Device {name} not found in scan.", flush=True)
                    continue

                success = await self._connect_and_subscribe(name, fresh_device)
                if success:
                    print(f"Reconnected to {name}.", flush=True)
                    self._disconnected_devices.pop(name, None)
                else:
                    print(f"Reconnect to {name} failed, will retry later.", flush=True)
            await asyncio.sleep(1.5)

    async def cleanup(self):
        try:
            await asyncio.gather(
                *(
                    c.stop_notify(self._char_uuid)
                    for c in self._connected_devices.values()
                )
            )
        except Exception as e:
            print(e, flush=True)

        for dev_name, dev in list(self._connected_devices.items()):
            try:
                await dev.disconnect()
            except Exception as e:
                print(f"Failed to disconnect {dev_name}", e, flush=True)
