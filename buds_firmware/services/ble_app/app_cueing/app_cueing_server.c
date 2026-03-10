/**
 ****************************************************************************************
 * @addtogroup APP
 * @{
 ****************************************************************************************
 */

#include "rwip_config.h"

#if (BLE_APP_CUEING_SERVER)

#include "app.h"
#include "app_cueing_server.h"
#include "app_task.h"
#include "arch.h"
#include "co_bt.h"
#include "cueingps_task.h"
#include "gapc_task.h"
#include "prf.h"
#include "prf_types.h"
#include "prf_utils.h"
#include "string.h"

#include "app_media_player.h"
#include "app_bt_stream.h"
#include "app_bt_media_manager.h"
#include "ke_timer.h"
#include "resources.h"

#define BLE_INVALID_CONNECTION_INDEX 0xFF
#define CUEING_VOLUME_SCALE_MAX 100
#define CUEING_HW_VOLUME_LEVELS 16

/*
 * GLOBAL VARIABLE DEFINITIONS
 ****************************************************************************************
 */

struct app_cueing_server_env_tag app_cueing_server_env = {
    BLE_INVALID_CONNECTION_INDEX,
    false,
    CUE_STATUS_IDLE,
    {0, 80, 500, 1, 0},
    0,
    false,
};

/*
 * LOCAL FUNCTIONS
 ****************************************************************************************
 */

static void cueing_send_status(uint8_t status_byte) {
  app_cueing_server_env.currentStatus = status_byte;

  if (app_cueing_server_env.isNotificationEnabled &&
      app_cueing_server_env.connectionIndex != BLE_INVALID_CONNECTION_INDEX) {

    struct ble_cueing_send_status_req_t *req = KE_MSG_ALLOC_DYN(
        CUEINGPS_SEND_STATUS_VIA_NOTIFICATION,
        prf_get_task_from_id(TASK_ID_CUEINGPS), TASK_APP,
        ble_cueing_send_status_req_t, 1);
    req->connectionIndex = app_cueing_server_env.connectionIndex;
    req->length = 1;
    req->value[0] = status_byte;
    ke_msg_send(req);
  }
}

static int8_t cueing_map_volume(uint8_t vol_pct) {
  if (vol_pct == 0)
    return TGT_VOLUME_LEVEL_MUTE;
  if (vol_pct >= CUEING_VOLUME_SCALE_MAX)
    return TGT_VOLUME_LEVEL_15;
  return (int8_t)(((uint32_t)vol_pct * CUEING_HW_VOLUME_LEVELS) /
                  CUEING_VOLUME_SCALE_MAX);
}

static AUD_ID_ENUM cueing_resolve_tone(uint8_t tone_id) {
  switch (tone_id) {
  case 0:
    return AUD_ID_BT_WARNING;
  case 1:
    return AUD_ID_NUM_1;
  case 2:
    return AUD_ID_NUM_2;
  case 3:
    return AUD_ID_NUM_3;
  case 4:
    return AUD_ID_NUM_4;
  default:
    return AUD_ID_BT_WARNING;
  }
}

static void cueing_cancel_timers(void) {
  ke_timer_clear(CUEINGPS_DURATION_TIMER, TASK_APP);
  ke_timer_clear(CUEINGPS_BURST_TIMER, TASK_APP);
}

static void cueing_play_single_tone(void) {
  AUD_ID_ENUM aud_id = cueing_resolve_tone(app_cueing_server_env.config.tone_id);
  app_bt_stream_volumeset(cueing_map_volume(app_cueing_server_env.config.volume));
  trigger_media_play(aud_id, 0, PROMOT_ID_BIT_MASK_CHNLSEl_ALL);
  app_cueing_server_env.tone_playing = true;
}

static void cueing_stop_single_tone(void) {
  if (app_cueing_server_env.tone_playing) {
    app_audio_manager_sendrequest(APP_BT_STREAM_MANAGER_STOP, BT_STREAM_MEDIA,
                                  0, 0);
    app_cueing_server_env.tone_playing = false;
  }
}

static void cueing_start_audio(void) {
  cueing_config_t *cfg = &app_cueing_server_env.config;

  TRACE(0, "CUEING: start tone=%d vol=%d dur=%d burst=%d gap=%d",
        cfg->tone_id, cfg->volume, cfg->duration_ms,
        cfg->burst_count, cfg->burst_gap_ms);

  cueing_cancel_timers();

  cueing_play_single_tone();
  cueing_send_status(CUE_STATUS_CUEING);

  if (cfg->burst_count > 1) {
    app_cueing_server_env.bursts_remaining = cfg->burst_count - 1;
    uint32_t tone_dur = cfg->duration_ms > 0 ? cfg->duration_ms : 200;
    ke_timer_set(CUEINGPS_BURST_TIMER, TASK_APP,
                 (tone_dur + cfg->burst_gap_ms) / 10);
  } else if (cfg->duration_ms > 0) {
    ke_timer_set(CUEINGPS_DURATION_TIMER, TASK_APP,
                 cfg->duration_ms / 10);
  }
}

static void cueing_stop_audio(void) {
  TRACE(0, "CUEING: stop audio cue");
  cueing_cancel_timers();
  app_cueing_server_env.bursts_remaining = 0;
  cueing_stop_single_tone();
  cueing_send_status(CUE_STATUS_IDLE);
}

static void cueing_handle_command(const uint8_t *data, uint16_t length) {
  if (length < 1) {
    return;
  }

  uint8_t cmd = data[0];

  switch (cmd) {
  case CUE_CMD_START:
    if (length >= 2) {
      app_cueing_server_env.config.tone_id = data[1];
    }
    if (length >= 3) {
      app_cueing_server_env.config.volume = data[2];
    }
    cueing_start_audio();
    break;

  case CUE_CMD_STOP:
    cueing_stop_audio();
    break;

  case CUE_CMD_CONFIGURE:
    if (length >= sizeof(cueing_config_t) + 1) {
      memcpy(&app_cueing_server_env.config, &data[1],
             sizeof(cueing_config_t));
      TRACE(0, "CUEING: config updated tone=%d vol=%d dur=%d burst=%d gap=%d",
            app_cueing_server_env.config.tone_id,
            app_cueing_server_env.config.volume,
            app_cueing_server_env.config.duration_ms,
            app_cueing_server_env.config.burst_count,
            app_cueing_server_env.config.burst_gap_ms);
    }
    cueing_send_status(CUE_STATUS_IDLE);
    break;

  default:
    TRACE(1, "CUEING: unknown command 0x%02x", cmd);
    cueing_send_status(CUE_STATUS_ERROR);
    break;
  }
}

/*
 * GLOBAL FUNCTIONS
 ****************************************************************************************
 */

void app_cueing_server_mtu_exchanged_handler(uint8_t conidx, uint16_t mtu) {
  (void)conidx;
  (void)mtu;
}

static void cueing_request_low_latency_params(uint8_t conidx) {
  struct gapc_conn_param conn_param;
  conn_param.intv_min = 6;    // 7.5ms
  conn_param.intv_max = 8;    // 10ms
  conn_param.latency = 0;
  conn_param.time_out = 200;  // 2s
  appm_update_param(conidx, &conn_param);
  TRACE(0, "CUEING: requested low-latency conn params (7.5-10ms)");
}

void app_cueing_server_connected_evt_handler(uint8_t conidx) {
  TRACE(0, "CUEING: connected conidx=%d", conidx);
  app_cueing_server_env.connectionIndex = conidx;
}

void app_cueing_server_disconnected_evt_handler(uint8_t conidx) {
  if (conidx == app_cueing_server_env.connectionIndex) {
    TRACE(0, "CUEING: disconnected");
    app_cueing_server_env.connectionIndex = BLE_INVALID_CONNECTION_INDEX;
    app_cueing_server_env.isNotificationEnabled = false;
    // Stop any active cueing on disconnect
    if (app_cueing_server_env.currentStatus == CUE_STATUS_CUEING) {
      cueing_stop_audio();
    }
  }
}

void app_cueing_server_init(void) {
  app_cueing_server_env.connectionIndex = BLE_INVALID_CONNECTION_INDEX;
  app_cueing_server_env.isNotificationEnabled = false;
  app_cueing_server_env.currentStatus = CUE_STATUS_IDLE;
  app_cueing_server_env.bursts_remaining = 0;
  app_cueing_server_env.tone_playing = false;
}

void app_cueing_add_cueingps(void) {
  TRACE(0, "CUEING: adding cueing profile service");
  struct gapm_profile_task_add_cmd *req =
      KE_MSG_ALLOC_DYN(GAPM_PROFILE_TASK_ADD_CMD, TASK_GAPM, TASK_APP,
                       gapm_profile_task_add_cmd, 0);

  req->operation = GAPM_PROFILE_TASK_ADD;
  req->sec_lvl = 0;
  req->prf_task_id = TASK_ID_CUEINGPS;
  req->app_task = TASK_APP;
  req->start_hdl = 0;

  ke_msg_send(req);
}

void app_cueing_server_send_status_notification(uint8_t *ptrData,
                                                uint32_t length) {
  struct ble_cueing_send_status_req_t *req = KE_MSG_ALLOC_DYN(
      CUEINGPS_SEND_STATUS_VIA_NOTIFICATION,
      prf_get_task_from_id(TASK_ID_CUEINGPS), TASK_APP,
      ble_cueing_send_status_req_t, length);
  req->connectionIndex = app_cueing_server_env.connectionIndex;
  req->length = length;
  memcpy(req->value, ptrData, length);
  ke_msg_send(req);
}

/*
 * MESSAGE HANDLERS
 ****************************************************************************************
 */

static int app_cueing_server_msg_handler(ke_msg_id_t const msgid,
                                         void const *param,
                                         ke_task_id_t const dest_id,
                                         ke_task_id_t const src_id) {
  return (KE_MSG_CONSUMED);
}

static int app_cueing_server_ntf_cfg_changed_handler(
    ke_msg_id_t const msgid, struct ble_cueing_status_ntf_cfg_t *param,
    ke_task_id_t const dest_id, ke_task_id_t const src_id) {

  app_cueing_server_env.isNotificationEnabled = param->isNotificationEnabled;

  if (app_cueing_server_env.isNotificationEnabled) {
    if (BLE_INVALID_CONNECTION_INDEX ==
        app_cueing_server_env.connectionIndex) {
      uint8_t conidx = KE_IDX_GET(src_id);
      app_cueing_server_connected_evt_handler(conidx);
    }
    cueing_request_low_latency_params(app_cueing_server_env.connectionIndex);
  }

  TRACE(1, "CUEING: notification %s",
        param->isNotificationEnabled ? "enabled" : "disabled");

  return (KE_MSG_CONSUMED);
}

static int app_cueing_server_cmd_received_handler(
    ke_msg_id_t const msgid, struct ble_cueing_cmd_ind_t *param,
    ke_task_id_t const dest_id, ke_task_id_t const src_id) {

  TRACE(1, "CUEING: received command, length=%d", param->length);
  cueing_handle_command(param->data, param->length);

  return (KE_MSG_CONSUMED);
}

static int app_cueing_server_tx_data_sent_handler(
    ke_msg_id_t const msgid, struct ble_cueing_tx_sent_ind_t *param,
    ke_task_id_t const dest_id, ke_task_id_t const src_id) {
  return (KE_MSG_CONSUMED);
}

static int app_cueing_duration_timer_handler(ke_msg_id_t const msgid,
                                             void const *param,
                                             ke_task_id_t const dest_id,
                                             ke_task_id_t const src_id) {
  TRACE(0, "CUEING: duration timer expired");
  cueing_stop_audio();
  return (KE_MSG_CONSUMED);
}

static int app_cueing_burst_timer_handler(ke_msg_id_t const msgid,
                                          void const *param,
                                          ke_task_id_t const dest_id,
                                          ke_task_id_t const src_id) {
  cueing_config_t *cfg = &app_cueing_server_env.config;

  cueing_stop_single_tone();

  if (app_cueing_server_env.bursts_remaining > 0) {
    app_cueing_server_env.bursts_remaining--;
    cueing_play_single_tone();

    if (app_cueing_server_env.bursts_remaining > 0) {
      uint32_t tone_dur = cfg->duration_ms > 0 ? cfg->duration_ms : 200;
      ke_timer_set(CUEINGPS_BURST_TIMER, TASK_APP,
                   (tone_dur + cfg->burst_gap_ms) / 10);
    } else if (cfg->duration_ms > 0) {
      ke_timer_set(CUEINGPS_DURATION_TIMER, TASK_APP,
                   cfg->duration_ms / 10);
    }
  } else {
    cueing_stop_audio();
  }

  return (KE_MSG_CONSUMED);
}

/*
 * LOCAL VARIABLE DEFINITIONS
 ****************************************************************************************
 */

const struct ke_msg_handler app_cueing_server_msg_handler_list[] = {
    {KE_MSG_DEFAULT_HANDLER, (ke_msg_func_t)app_cueing_server_msg_handler},
    {CUEINGPS_STATUS_NTF_CFG_CHANGED,
     (ke_msg_func_t)app_cueing_server_ntf_cfg_changed_handler},
    {CUEINGPS_TX_DATA_SENT,
     (ke_msg_func_t)app_cueing_server_tx_data_sent_handler},
    {CUEINGPS_CUE_CMD_RECEIVED,
     (ke_msg_func_t)app_cueing_server_cmd_received_handler},
    {CUEINGPS_DURATION_TIMER,
     (ke_msg_func_t)app_cueing_duration_timer_handler},
    {CUEINGPS_BURST_TIMER,
     (ke_msg_func_t)app_cueing_burst_timer_handler},
};

const struct ke_state_handler app_cueing_server_table_handler = {
    &app_cueing_server_msg_handler_list[0],
    (sizeof(app_cueing_server_msg_handler_list) /
     sizeof(struct ke_msg_handler))};

#endif // BLE_APP_CUEING_SERVER

/// @} APP
