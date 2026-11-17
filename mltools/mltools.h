// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    mltools.h
 * @brief   Public header for the milk-mltools L1 module (`mlt`)
 */

#ifndef MILK_MLTOOLS_H
#define MILK_MLTOOLS_H

#include "milk_runnable.h"
#include "tlm_fileset.h"

extern const MILK_RUNNABLE runnable_mlt_tlmindex;
extern const MILK_RUNNABLE runnable_mlt_tlmload;
extern const MILK_RUNNABLE runnable_mlt_tlmsync;
extern const MILK_RUNNABLE runnable_mlt_tlmsave;

#endif /* MILK_MLTOOLS_H */
