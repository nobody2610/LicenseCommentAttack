/******************************************************************************
    Copyright (C) 2023 by Lain Bailey <lain@obsproject.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
******************************************************************************/
#pragma once

#include "util/c99defs.h"
#include "util/bmem.h"
#include "util/profiler.h"
#include "util/text-lookup.h"
#include "graphics/graphics.h"
#include "graphics/vec2.h"
#include "graphics/vec3.h"
#include "media-io/audio-io.h"
#include "media-io/video-io.h"
#include "callback/signal.h"
#include "callback/proc.h"

#include "obs-config.h"
#include "obs-defs.h"
#include "obs-data.h"
#include "obs-properties.h"
#include "obs-interaction.h"

struct matrix4;

/* opaque types */
struct obs_context_data;
struct obs_display;
struct obs_view;
struct obs_source;
struct obs_scene;
struct obs_scene_item;
struct obs_output;
struct obs_encoder;
struct obs_encoder_group;
struct obs_service;
struct obs_module;
struct obs_module_metadata;
struct obs_fader;
struct obs_volmeter;
struct obs_canvas;

typedef struct obs_context_data obs_object_t;
typedef struct obs_display obs_display_t;
typedef struct obs_view obs_view_t;
typedef struct obs_source obs_source_t;
typedef struct obs_scene obs_scene_t;
typedef struct obs_scene_item obs_sceneitem_t;
typedef struct obs_output obs_output_t;
typedef struct obs_encoder obs_encoder_t;
typedef struct obs_encoder_group obs_encoder_group_t;
typedef struct obs_service obs_service_t;
typedef struct obs_module obs_module_t;
typedef struct obs_module_metadata obs_module_metadata_t;
typedef struct obs_fader obs_fader_t;
typedef struct obs_volmeter obs_volmeter_t;
typedef struct obs_canvas obs_canvas_t;

typedef struct obs_weak_object obs_weak_object_t;
typedef struct obs_weak_source obs_weak_source_t;
typedef struct obs_weak_output obs_weak_output_t;
typedef struct obs_weak_encoder obs_weak_encoder_t;
typedef struct obs_weak_service obs_weak_service_t;
typedef struct obs_weak_canvas obs_weak_canvas_t;

#include "obs-missing-files.h"
#include "obs-source.h"
#include "obs-encoder.h"
#include "obs-output.h"
#include "obs-service.h"
#include "obs-audio-controls.h"
#include "obs-hotkey.h"

/**
 * @file
 * @brief Main libobs header used by applications.
 *
 * @mainpage
 *
 * @section intro_sec Introduction
 *
 * This document describes the api for libobs to be used by applications as well
 * as @ref modules_page implementing some kind of functionality.
 *
 */

#ifdef __cplusplus
extern "C" {
#endif

/** Used for changing the order of items (for example, filters in a source,
 * or items in a scene) */
enum obs_order_movement {
	OBS_ORDER_MOVE_UP,
	OBS_ORDER_MOVE_DOWN,
	OBS_ORDER_MOVE_TOP,
	OBS_ORDER_MOVE_BOTTOM,
};

/**
 * Used with obs_source_process_filter to specify whether the filter should
 * render the source directly with the specified effect, or whether it should
 * render it to a texture
 */
enum obs_allow_direct_render {
	OBS_NO_DIRECT_RENDERING,
	OBS_ALLOW_DIRECT_RENDERING,
};

enum obs_scale_type {
	OBS_SCALE_DISABLE,
	OBS_SCALE_POINT,
	OBS_SCALE_BICUBIC,
	OBS_SCALE_BILINEAR,
	OBS_SCALE_LANCZOS,
	OBS_SCALE_AREA,
};

enum obs_blending_method {
	OBS_BLEND_METHOD_DEFAULT,
	OBS_BLEND_METHOD_SRGB_OFF,
};

enum obs_blending_type {
	OBS_BLEND_NORMAL,
	OBS_BLEND_ADDITIVE,
	OBS_BLEND_SUBTRACT,
	OBS_BLEND_SCREEN,
	OBS_BLEND_MULTIPLY,
	OBS_BLEND_LIGHTEN,
	OBS_BLEND_DARKEN,
};

/**
 * Used with scene items to indicate the type of bounds to use for scene items.
 * Mostly determines how the image will be scaled within those bounds, or
 * whether to use bounds at all.
 */
enum obs_bounds_type {
	OBS_BOUNDS_NONE,            /**< no bounds */
	OBS_BOUNDS_STRETCH,         /**< stretch (ignores base scale) */
	OBS_BOUNDS_SCALE_INNER,     /**< scales to inner rectangle */
	OBS_BOUNDS_SCALE_OUTER,     /**< scales to outer rectangle */
	OBS_BOUNDS_SCALE_TO_WIDTH,  /**< scales to the width  */
	OBS_BOUNDS_SCALE_TO_HEIGHT, /**< scales to the height */
	OBS_BOUNDS_MAX_ONLY,        /**< no scaling, maximum size only */
};

/**
 * Used by libobs to define the state of a plugin/module.
 */
enum obs_module_load_state {
	OBS_MODULE_INVALID,
	OBS_MODULE_ENABLED,
	OBS_MODULE_MISSING,
	OBS_MODULE_DISABLED,
	OBS_MODULE_DISABLED_SAFE,