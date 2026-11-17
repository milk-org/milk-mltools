// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    mltools.c
 * @brief   Module registration for milk-mltools (`mlt.*`)
 */

#define MODULE_SHORTNAME_DEFAULT "mlt"
#define MODULE_DESCRIPTION "Machine learning and time-series telemetry tools"

#include "milk_runnable.h"
#include "mltools.h"

/* CFITSIO-mandated runnables are weak so builds without CFITSIO still load the module. */
MILK_RUNNABLE_WEAK_DECL(runnable_mlt_tlmload);
MILK_RUNNABLE_WEAK_DECL(runnable_mlt_tlmsave);

static const MILK_RUNNABLE *const runnables[] = {
    &runnable_mlt_tlmindex,
    &runnable_mlt_tlmload,
    &runnable_mlt_tlmsync,
    &runnable_mlt_tlmsave,
};

MILK_MODULE_RUNNABLES(milkmltools, NULL, runnables);
