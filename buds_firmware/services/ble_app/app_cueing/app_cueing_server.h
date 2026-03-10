#ifndef APP_CUEING_SERVER_H_
#define APP_CUEING_SERVER_H_

/**
 ****************************************************************************************
 * @addtogroup APP
 * @brief Audio Cueing Server Application entry point.
 * @{
 ****************************************************************************************
 */

#include "rwip_config.h"

#if (BLE_APP_CUEING_SERVER)

#include <stdint.h>
#include "ke_task.h"

/*
 * DEFINES
 ****************************************************************************************
 */

/* Command bytes sent by the host via the Cue Command characteristic */
#define CUE_CMD_START     0x01
#define CUE_CMD_STOP      0x02
#define CUE_CMD_CONFIGURE 0x03

/* Status bytes sent back via notification on the Cue Status characteristic */
#define CUE_STATUS_IDLE   0x00
#define CUE_STATUS_CUEING 0x01
#define CUE_STATUS_ERROR  0xFF

/*
 * TYPE DEFINITIONS
 ****************************************************************************************
 */

/// Cueing configuration (written via Cue Config or payload of CUE_CMD_CONFIGURE)
typedef struct {
  uint8_t tone_id;       // 0=beep, 1=click, 2=chirp
  uint8_t volume;        // 0-100
  uint16_t duration_ms;  // 0 = until stopped
  uint8_t burst_count;   // number of pulses, 0 = continuous
  uint16_t burst_gap_ms; // gap between pulses
} __attribute__((packed)) cueing_config_t;

struct app_cueing_server_env_tag {
  uint8_t connectionIndex;
  uint8_t isNotificationEnabled;
  uint8_t currentStatus;
  cueing_config_t config;
  uint8_t bursts_remaining;
  bool tone_playing;
};

/*
 * GLOBAL VARIABLES DECLARATIONS
 ****************************************************************************************
 */

extern struct app_cueing_server_env_tag app_cueing_server_env;
extern const struct ke_state_handler app_cueing_server_table_handler;

#ifdef __cplusplus
extern "C" {
#endif

void app_cueing_server_init(void);
void app_cueing_add_cueingps(void);
void app_cueing_server_connected_evt_handler(uint8_t conidx);
void app_cueing_server_disconnected_evt_handler(uint8_t conidx);
void app_cueing_server_send_status_notification(uint8_t *ptrData,
                                                uint32_t length);
void app_cueing_server_mtu_exchanged_handler(uint8_t conidx, uint16_t mtu);

#ifdef __cplusplus
}
#endif

#endif // (BLE_APP_CUEING_SERVER)

/// @} APP

#endif // APP_CUEING_SERVER_H_
