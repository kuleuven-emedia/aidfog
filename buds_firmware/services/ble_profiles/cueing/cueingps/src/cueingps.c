/**
 ****************************************************************************************
 * @addtogroup CUEINGPS
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

/*
 * CUEING SERVICE PROFILE ATTRIBUTES
 ****************************************************************************************
 */

/*
 * Custom 128-bit UUIDs for the Audio Cueing Service.
 * Base UUID: AC0000xx-CAFE-B0BA-F0G1-DEADBEEF0000
 * Service:   AC000001-...
 * Cmd Char:  AC000002-...
 * Status:    AC000003-...
 * Config:    AC000004-...
 */
#define cueing_service_uuid_128_content                                         \
  {                                                                            \
    0x00, 0x00, 0xEF, 0xBE, 0xAD, 0xDE, 0x01, 0xF0, 0xBA, 0xB0, 0xFE, 0xCA, \
        0x01, 0x00, 0x00, 0xAC                                                \
  }

#define cueing_cmd_char_uuid_128_content                                       \
  {                                                                            \
    0x00, 0x00, 0xEF, 0xBE, 0xAD, 0xDE, 0x01, 0xF0, 0xBA, 0xB0, 0xFE, 0xCA, \
        0x02, 0x00, 0x00, 0xAC                                                \
  }

#define cueing_status_char_uuid_128_content                                    \
  {                                                                            \
    0x00, 0x00, 0xEF, 0xBE, 0xAD, 0xDE, 0x01, 0xF0, 0xBA, 0xB0, 0xFE, 0xCA, \
        0x03, 0x00, 0x00, 0xAC                                                \
  }

#define cueing_config_char_uuid_128_content                                    \
  {                                                                            \
    0x00, 0x00, 0xEF, 0xBE, 0xAD, 0xDE, 0x01, 0xF0, 0xBA, 0xB0, 0xFE, 0xCA, \
        0x04, 0x00, 0x00, 0xAC                                                \
  }

#define ATT_DECL_PRIMARY_SERVICE_UUID                                          \
  { 0x00, 0x28 }
#define ATT_DECL_CHARACTERISTIC_UUID                                           \
  { 0x03, 0x28 }
#define ATT_DESC_CLIENT_CHAR_CFG_UUID                                          \
  { 0x02, 0x29 }
#define ATT_DESC_CHAR_USER_DESCRIPTION_UUID                                    \
  { 0x01, 0x29 }

static const uint8_t CUEING_SERVICE_UUID_128[ATT_UUID_128_LEN] =
    cueing_service_uuid_128_content;

/// Full CUEING SERVER Database Description
const struct attm_desc_128 cueingps_att_db[CUEINGPS_IDX_NB] = {
    // Service Declaration
    [CUEINGPS_IDX_SVC] = {ATT_DECL_PRIMARY_SERVICE_UUID, PERM(RD, ENABLE), 0,
                          0},

    // Cue Command Characteristic Declaration
    [CUEINGPS_IDX_CMD_CHAR] = {ATT_DECL_CHARACTERISTIC_UUID, PERM(RD, ENABLE),
                               0, 0},
    // Cue Command Value (Write, Write Without Response)
    [CUEINGPS_IDX_CMD_VAL] = {cueing_cmd_char_uuid_128_content,
                              PERM(WRITE_REQ, ENABLE) |
                                  PERM(WRITE_COMMAND, ENABLE),
                              PERM(RI, ENABLE) |
                                  PERM_VAL(UUID_LEN, PERM_UUID_128),
                              CUEINGPS_MAX_LEN},
    // Cue Command User Description
    [CUEINGPS_IDX_CMD_DESC] = {ATT_DESC_CHAR_USER_DESCRIPTION_UUID,
                               PERM(RD, ENABLE), PERM(RI, ENABLE), 32},

    // Cue Status Characteristic Declaration
    [CUEINGPS_IDX_STATUS_CHAR] = {ATT_DECL_CHARACTERISTIC_UUID,
                                  PERM(RD, ENABLE), 0, 0},
    // Cue Status Value (Read + Notify)
    [CUEINGPS_IDX_STATUS_VAL] = {cueing_status_char_uuid_128_content,
                                 PERM(NTF, ENABLE) | PERM(RD, ENABLE),
                                 PERM(RI, ENABLE) |
                                     PERM_VAL(UUID_LEN, PERM_UUID_128),
                                 CUEINGPS_MAX_LEN},
    // Cue Status CCCD (for notifications)
    [CUEINGPS_IDX_STATUS_NTF_CFG] = {ATT_DESC_CLIENT_CHAR_CFG_UUID,
                                     PERM(RD, ENABLE) |
                                         PERM(WRITE_REQ, ENABLE),
                                     0, 0},
    // Cue Status User Description
    [CUEINGPS_IDX_STATUS_DESC] = {ATT_DESC_CHAR_USER_DESCRIPTION_UUID,
                                  PERM(RD, ENABLE), PERM(RI, ENABLE), 32},

    // Cue Config Characteristic Declaration
    [CUEINGPS_IDX_CONFIG_CHAR] = {ATT_DECL_CHARACTERISTIC_UUID,
                                  PERM(RD, ENABLE), 0, 0},
    // Cue Config Value (Read + Write)
    [CUEINGPS_IDX_CONFIG_VAL] = {cueing_config_char_uuid_128_content,
                                 PERM(RD, ENABLE) | PERM(WRITE_REQ, ENABLE),
                                 PERM(RI, ENABLE) |
                                     PERM_VAL(UUID_LEN, PERM_UUID_128),
                                 CUEINGPS_MAX_LEN},
    // Cue Config User Description
    [CUEINGPS_IDX_CONFIG_DESC] = {ATT_DESC_CHAR_USER_DESCRIPTION_UUID,
                                  PERM(RD, ENABLE), PERM(RI, ENABLE), 32},
};

/**
 ****************************************************************************************
 * @brief Initialization of the CUEINGPS module.
 ****************************************************************************************
 */
static uint8_t cueingps_init(struct prf_task_env *env, uint16_t *start_hdl,
                             uint16_t app_task, uint8_t sec_lvl,
                             void *params) {
  uint8_t status;

  status = attm_svc_create_db_128(
      start_hdl, CUEING_SERVICE_UUID_128, NULL, CUEINGPS_IDX_NB, NULL,
      env->task, &cueingps_att_db[0],
      (sec_lvl &
       (PERM_MASK_SVC_DIS | PERM_MASK_SVC_AUTH | PERM_MASK_SVC_EKS)) |
          PERM(SVC_MI, DISABLE) | PERM_VAL(SVC_UUID_LEN, PERM_UUID_128));

  if (status == ATT_ERR_NO_ERROR) {
    struct cueingps_env_tag *cueingps_env =
        (struct cueingps_env_tag *)ke_malloc(sizeof(struct cueingps_env_tag),
                                            KE_MEM_ATT_DB);

    memset((uint8_t *)cueingps_env, 0, sizeof(struct cueingps_env_tag));

    env->env = (prf_env_t *)cueingps_env;
    cueingps_env->shdl = *start_hdl;

    cueingps_env->prf_env.app_task =
        app_task | (PERM_GET(sec_lvl, SVC_MI) ? PERM(PRF_MI, ENABLE)
                                              : PERM(PRF_MI, DISABLE));
    cueingps_env->prf_env.prf_task = env->task | PERM(PRF_MI, DISABLE);

    env->id = TASK_ID_CUEINGPS;
    cueingps_task_init(&(env->desc));

    ke_state_set(env->task, CUEINGPS_IDLE);
  }

  return (status);
}

static void cueingps_destroy(struct prf_task_env *env) {
  struct cueingps_env_tag *cueingps_env =
      (struct cueingps_env_tag *)env->env;
  env->env = NULL;
  ke_free(cueingps_env);
}

static void cueingps_create(struct prf_task_env *env, uint8_t conidx) {
  struct cueingps_env_tag *cueingps_env =
      (struct cueingps_env_tag *)env->env;
  struct prf_svc cueingps_svc = {cueingps_env->shdl,
                                 cueingps_env->shdl + CUEINGPS_IDX_NB};
  prf_register_atthdl2gatt(env->env, conidx, &cueingps_svc);
}

static void cueingps_cleanup(struct prf_task_env *env, uint8_t conidx,
                             uint8_t reason) {
  /* Nothing to do */
}

/*
 * GLOBAL VARIABLE DEFINITIONS
 ****************************************************************************************
 */

const struct prf_task_cbs cueingps_itf = {
    (prf_init_fnct)cueingps_init,
    cueingps_destroy,
    cueingps_create,
    cueingps_cleanup,
};

const struct prf_task_cbs *cueingps_prf_itf_get(void) {
  return &cueingps_itf;
}

#endif /* BLE_CUEING_SERVER */

/// @} CUEINGPS
