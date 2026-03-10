#ifndef _CUEINGPS_H_
#define _CUEINGPS_H_

/**
 ****************************************************************************************
 * @addtogroup CUEINGPS Audio Cueing Profile Server
 * @ingroup CUEING
 * @brief Audio Cueing Profile Server
 *
 * @{
 ****************************************************************************************
 */

#include "rwip_config.h"

#if (BLE_CUEING_SERVER)
#include "attm.h"
#include "cueingps_task.h"
#include "prf.h"
#include "prf_types.h"
#include "prf_utils.h"

#define CUEINGPS_MAX_LEN (20)

static const char cueing_cmd_desc[] = "Cue Command";
static const char cueing_status_desc[] = "Cue Status";
static const char cueing_config_desc[] = "Cue Config";

/*
 * DEFINES
 ****************************************************************************************
 */

/// Possible states of the CUEINGPS task
enum {
  CUEINGPS_IDLE,
  CUEINGPS_BUSY,
  CUEINGPS_STATE_MAX,
};

/// Attributes State Machine
enum {
  CUEINGPS_IDX_SVC,

  CUEINGPS_IDX_CMD_CHAR,
  CUEINGPS_IDX_CMD_VAL,
  CUEINGPS_IDX_CMD_DESC,

  CUEINGPS_IDX_STATUS_CHAR,
  CUEINGPS_IDX_STATUS_VAL,
  CUEINGPS_IDX_STATUS_NTF_CFG,
  CUEINGPS_IDX_STATUS_DESC,

  CUEINGPS_IDX_CONFIG_CHAR,
  CUEINGPS_IDX_CONFIG_VAL,
  CUEINGPS_IDX_CONFIG_DESC,

  CUEINGPS_IDX_NB,
};

/*
 * TYPE DEFINITIONS
 ****************************************************************************************
 */

/// Audio Cueing Profile Server environment variable
struct cueingps_env_tag {
  prf_env_t prf_env;
  uint16_t shdl;
  uint8_t isNotificationEnabled[BLE_CONNECTION_MAX];
  ke_state_t state;
};

/*
 * FUNCTION DECLARATIONS
 ****************************************************************************************
 */

const struct prf_task_cbs *cueingps_prf_itf_get(void);
void cueingps_task_init(struct ke_task_desc *task_desc);

#endif /* #if (BLE_CUEING_SERVER) */

/// @} CUEINGPS

#endif /* _CUEINGPS_H_ */
