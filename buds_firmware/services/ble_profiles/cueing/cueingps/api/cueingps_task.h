#ifndef _CUEINGPS_TASK_H_
#define _CUEINGPS_TASK_H_

/**
 ****************************************************************************************
 * @addtogroup CUEINGPSTASK Task
 * @ingroup CUEINGPS
 * @brief Audio Cueing Profile Server Task
 *
 * @{
 ****************************************************************************************
 */

#include <stdint.h>
#include "rwip_task.h"

/*
 * DEFINES
 ****************************************************************************************
 */

/// Messages for Audio Cueing Server Profile
enum cueingps_msg_id {
  CUEINGPS_CUE_CMD_RECEIVED = TASK_FIRST_MSG(TASK_ID_CUEINGPS),
  CUEINGPS_STATUS_NTF_CFG_CHANGED,
  CUEINGPS_TX_DATA_SENT,
  CUEINGPS_SEND_STATUS_VIA_NOTIFICATION,
  CUEINGPS_DURATION_TIMER,
  CUEINGPS_BURST_TIMER,
};

/*
 * TYPE DEFINITIONS
 ****************************************************************************************
 */

struct ble_cueing_cmd_ind_t {
  uint16_t length;
  uint8_t data[0];
};

struct ble_cueing_status_ntf_cfg_t {
  bool isNotificationEnabled;
};

struct ble_cueing_tx_sent_ind_t {
  uint8_t status;
};

struct ble_cueing_send_status_req_t {
  uint8_t connectionIndex;
  uint32_t length;
  uint8_t value[__ARRAY_EMPTY];
};

/// @} CUEINGPSTASK

#endif /* _CUEINGPS_TASK_H_ */
