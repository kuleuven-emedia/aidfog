/**
 ****************************************************************************************
 * @addtogroup CUEINGPSTASK
 * @{
 ****************************************************************************************
 */

#include "rwip_config.h"

#if (BLE_CUEING_SERVER)
#include "attm.h"
#include "cueingps.h"
#include "cueingps_task.h"
#include "gap.h"
#include "gapc_task.h"
#include "gattc_task.h"

#include "prf_utils.h"

#include "co_utils.h"
#include "ke_mem.h"
#include "app_cueing_server.h"

static int gapc_disconnect_ind_handler(ke_msg_id_t const msgid,
                                       struct gapc_disconnect_ind const *param,
                                       ke_task_id_t const dest_id,
                                       ke_task_id_t const src_id) {
  struct cueingps_env_tag *cueingps_env =
      PRF_ENV_GET(CUEINGPS, cueingps);
  uint8_t conidx = KE_IDX_GET(src_id);
  cueingps_env->isNotificationEnabled[conidx] = false;
  return KE_MSG_CONSUMED;
}

/**
 ****************************************************************************************
 * @brief Handles write request from peer device.
 ****************************************************************************************
 */
__STATIC int gattc_write_req_ind_handler(
    ke_msg_id_t const msgid, struct gattc_write_req_ind const *param,
    ke_task_id_t const dest_id, ke_task_id_t const src_id) {

  struct cueingps_env_tag *cueingps_env =
      PRF_ENV_GET(CUEINGPS, cueingps);
  uint8_t conidx = KE_IDX_GET(src_id);
  uint8_t status = GAP_ERR_NO_ERROR;

  if (cueingps_env != NULL) {
    // Status CCCD write (enable/disable notifications)
    if (param->handle ==
        (cueingps_env->shdl + CUEINGPS_IDX_STATUS_NTF_CFG)) {
      uint16_t value = 0x0000;
      memcpy(&value, &(param->value), sizeof(uint16_t));

      if (value == PRF_CLI_STOP_NTFIND) {
        cueingps_env->isNotificationEnabled[conidx] = false;
      } else if (value == PRF_CLI_START_NTF) {
        cueingps_env->isNotificationEnabled[conidx] = true;
      } else {
        status = PRF_APP_ERROR;
      }

      if (status == GAP_ERR_NO_ERROR) {
        struct ble_cueing_status_ntf_cfg_t *ind = KE_MSG_ALLOC(
            CUEINGPS_STATUS_NTF_CFG_CHANGED,
            prf_dst_task_get(&cueingps_env->prf_env, conidx),
            prf_src_task_get(&cueingps_env->prf_env, conidx),
            ble_cueing_status_ntf_cfg_t);

        ind->isNotificationEnabled =
            cueingps_env->isNotificationEnabled[conidx];

        ke_msg_send(ind);
      }
    }
    // Cue Command write
    else if (param->handle ==
             (cueingps_env->shdl + CUEINGPS_IDX_CMD_VAL)) {
      struct ble_cueing_cmd_ind_t *ind = KE_MSG_ALLOC_DYN(
          CUEINGPS_CUE_CMD_RECEIVED,
          prf_dst_task_get(&cueingps_env->prf_env, conidx),
          prf_src_task_get(&cueingps_env->prf_env, conidx),
          ble_cueing_cmd_ind_t, param->length);

      ind->length = param->length;
      memcpy((uint8_t *)(ind->data), &(param->value), param->length);

      ke_msg_send(ind);
    }
    // Cue Config write: prepend CUE_CMD_CONFIGURE so the app layer
    // processes it through the same command path
    else if (param->handle ==
             (cueingps_env->shdl + CUEINGPS_IDX_CONFIG_VAL)) {
      struct ble_cueing_cmd_ind_t *ind = KE_MSG_ALLOC_DYN(
          CUEINGPS_CUE_CMD_RECEIVED,
          prf_dst_task_get(&cueingps_env->prf_env, conidx),
          prf_src_task_get(&cueingps_env->prf_env, conidx),
          ble_cueing_cmd_ind_t, param->length + 1);

      ind->length = param->length + 1;
      ind->data[0] = 0x03;  // CUE_CMD_CONFIGURE
      memcpy((uint8_t *)(ind->data + 1), &(param->value), param->length);

      ke_msg_send(ind);
    } else {
      status = PRF_APP_ERROR;
    }
  }

  // Send write response
  struct gattc_write_cfm *cfm =
      KE_MSG_ALLOC(GATTC_WRITE_CFM, src_id, dest_id, gattc_write_cfm);
  cfm->handle = param->handle;
  cfm->status = status;
  ke_msg_send(cfm);

  return (KE_MSG_CONSUMED);
}

__STATIC int gattc_cmp_evt_handler(ke_msg_id_t const msgid,
                                   struct gattc_cmp_evt const *param,
                                   ke_task_id_t const dest_id,
                                   ke_task_id_t const src_id) {
  struct cueingps_env_tag *cueingps_env =
      PRF_ENV_GET(CUEINGPS, cueingps);
  uint8_t conidx = KE_IDX_GET(dest_id);

  if (GATTC_NOTIFY == param->operation) {
    struct ble_cueing_tx_sent_ind_t *ind = KE_MSG_ALLOC(
        CUEINGPS_TX_DATA_SENT,
        prf_dst_task_get(&cueingps_env->prf_env, conidx),
        prf_src_task_get(&cueingps_env->prf_env, conidx),
        ble_cueing_tx_sent_ind_t);

    ind->status = param->status;
    ke_msg_send(ind);
  }

  ke_state_set(dest_id, CUEINGPS_IDLE);

  return (KE_MSG_CONSUMED);
}

__STATIC int gattc_read_req_ind_handler(
    ke_msg_id_t const msgid, struct gattc_read_req_ind const *param,
    ke_task_id_t const dest_id, ke_task_id_t const src_id) {

  struct cueingps_env_tag *cueingps_env =
      PRF_ENV_GET(CUEINGPS, cueingps);

  struct gattc_read_cfm *cfm =
      KE_MSG_ALLOC_DYN(GATTC_READ_CFM, src_id, dest_id, gattc_read_cfm, 32);

  uint8_t conidx = KE_IDX_GET(src_id);
  uint8_t status = GAP_ERR_NO_ERROR;

  if (param->handle == (cueingps_env->shdl + CUEINGPS_IDX_CMD_DESC)) {
    cfm->length = sizeof(cueing_cmd_desc) - 1;
    memcpy(cfm->value, cueing_cmd_desc, cfm->length);
  } else if (param->handle ==
             (cueingps_env->shdl + CUEINGPS_IDX_STATUS_DESC)) {
    cfm->length = sizeof(cueing_status_desc) - 1;
    memcpy(cfm->value, cueing_status_desc, cfm->length);
  } else if (param->handle ==
             (cueingps_env->shdl + CUEINGPS_IDX_STATUS_NTF_CFG)) {
    uint16_t notify_ccc =
        cueingps_env->isNotificationEnabled[conidx] ? 1 : 0;
    cfm->length = sizeof(notify_ccc);
    memcpy(cfm->value, (uint8_t *)&notify_ccc, cfm->length);
  } else if (param->handle ==
             (cueingps_env->shdl + CUEINGPS_IDX_CONFIG_DESC)) {
    cfm->length = sizeof(cueing_config_desc) - 1;
    memcpy(cfm->value, cueing_config_desc, cfm->length);
  } else if (param->handle ==
             (cueingps_env->shdl + CUEINGPS_IDX_STATUS_VAL)) {
    cfm->length = 1;
    cfm->value[0] = app_cueing_server_env.currentStatus;
  } else if (param->handle ==
             (cueingps_env->shdl + CUEINGPS_IDX_CONFIG_VAL)) {
    cfm->length = sizeof(cueing_config_t);
    memcpy(cfm->value, &app_cueing_server_env.config,
           sizeof(cueing_config_t));
  } else {
    cfm->length = 0;
    status = ATT_ERR_REQUEST_NOT_SUPPORTED;
  }

  cfm->handle = param->handle;
  cfm->status = status;
  ke_msg_send(cfm);

  ke_state_set(dest_id, CUEINGPS_IDLE);

  return (KE_MSG_CONSUMED);
}

static void send_status_notification(uint8_t conidx, const uint8_t *ptrData,
                                     uint32_t length) {
  struct cueingps_env_tag *cueingps_env =
      PRF_ENV_GET(CUEINGPS, cueingps);

  if (cueingps_env->isNotificationEnabled[conidx]) {
    struct gattc_send_evt_cmd *report_ntf = KE_MSG_ALLOC_DYN(
        GATTC_SEND_EVT_CMD, KE_BUILD_ID(TASK_GATTC, conidx),
        prf_src_task_get(&cueingps_env->prf_env, conidx),
        gattc_send_evt_cmd, length);

    report_ntf->operation = GATTC_NOTIFY;
    report_ntf->handle = cueingps_env->shdl + CUEINGPS_IDX_STATUS_VAL;
    report_ntf->length = length;
    memcpy(report_ntf->value, ptrData, length);
    ke_msg_send(report_ntf);
  }
}

__STATIC int send_status_via_notification_handler(
    ke_msg_id_t const msgid,
    struct ble_cueing_send_status_req_t const *param,
    ke_task_id_t const dest_id, ke_task_id_t const src_id) {
  send_status_notification(param->connectionIndex, param->value, param->length);
  return (KE_MSG_CONSUMED);
}

static int gattc_att_info_req_ind_handler(
    ke_msg_id_t const msgid, struct gattc_att_info_req_ind *param,
    ke_task_id_t const dest_id, ke_task_id_t const src_id) {

  struct gattc_att_info_cfm *cfm;
  cfm = KE_MSG_ALLOC(GATTC_ATT_INFO_CFM, src_id, dest_id, gattc_att_info_cfm);
  cfm->handle = param->handle;

  struct cueingps_env_tag *cueingps_env =
      PRF_ENV_GET(CUEINGPS, cueingps);

  if (param->handle == cueingps_env->shdl + CUEINGPS_IDX_STATUS_NTF_CFG) {
    cfm->length = 2;
    cfm->status = GAP_ERR_NO_ERROR;
  } else if (param->handle == cueingps_env->shdl + CUEINGPS_IDX_CMD_VAL) {
    cfm->length = 0;
    cfm->status = GAP_ERR_NO_ERROR;
  } else if (param->handle == cueingps_env->shdl + CUEINGPS_IDX_CONFIG_VAL) {
    cfm->length = 0;
    cfm->status = GAP_ERR_NO_ERROR;
  } else {
    cfm->length = 0;
    cfm->status = ATT_ERR_WRITE_NOT_PERMITTED;
  }
  ke_msg_send(cfm);

  return (KE_MSG_CONSUMED);
}

/*
 * GLOBAL VARIABLE DEFINITIONS
 ****************************************************************************************
 */

KE_MSG_HANDLER_TAB(cueingps){
    {GAPC_DISCONNECT_IND, (ke_msg_func_t)gapc_disconnect_ind_handler},
    {GATTC_WRITE_REQ_IND, (ke_msg_func_t)gattc_write_req_ind_handler},
    {GATTC_CMP_EVT, (ke_msg_func_t)gattc_cmp_evt_handler},
    {GATTC_READ_REQ_IND, (ke_msg_func_t)gattc_read_req_ind_handler},
    {CUEINGPS_SEND_STATUS_VIA_NOTIFICATION,
     (ke_msg_func_t)send_status_via_notification_handler},
    {GATTC_ATT_INFO_REQ_IND, (ke_msg_func_t)gattc_att_info_req_ind_handler},
};

void cueingps_task_init(struct ke_task_desc *task_desc) {
  struct cueingps_env_tag *cueingps_env =
      PRF_ENV_GET(CUEINGPS, cueingps);

  task_desc->msg_handler_tab = cueingps_msg_handler_tab;
  task_desc->msg_cnt = ARRAY_LEN(cueingps_msg_handler_tab);
  task_desc->state = &(cueingps_env->state);
  task_desc->idx_max = BLE_CONNECTION_MAX;
}

#endif /* #if (BLE_CUEING_SERVER) */

/// @} CUEINGPSTASK
