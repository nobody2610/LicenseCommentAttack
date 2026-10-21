/**
 * \file include/pcm.h
 * \brief Application interface library for the ALSA driver
 * \author Jaroslav Kysela <perex@perex.cz>
 * \author Abramo Bagnara <abramo@alsa-project.org>
 * \author Takashi Iwai <tiwai@suse.de>
 * \date 1998-2001
 *
 * Application interface library for the ALSA driver.
 * See the \ref pcm page for more details.
 */
/*
 *   This library is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU Lesser General Public License as
 *   published by the Free Software Foundation; either version 2.1 of
 *   the License, or (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU Lesser General Public License for more details.
 *
 *   You should have received a copy of the GNU Lesser General Public
 *   License along with this library; if not, write to the Free Software
 *   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

#ifndef __ALSA_PCM_H
#define __ALSA_PCM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/**
 *  \defgroup PCM PCM Interface
 *  See the \ref pcm page for more details.
 *  \{
 */

/** dlsym version for interface entry callback */
#define SND_PCM_DLSYM_VERSION		_dlsym_pcm_001

/** PCM generic info container */
typedef struct _snd_pcm_info snd_pcm_info_t;

/** PCM hardware configuration space container
 *
 *  snd_pcm_hw_params_t is an opaque structure which contains a set of possible
 *  PCM hardware configurations. For example, a given instance might include a
 *  range of buffer sizes, a range of period sizes, and a set of several sample
 *  formats. Some subset of all possible combinations these sets may be valid,
 *  but not necessarily any combination will be valid.
 *
 *  When a parameter is set or restricted using a snd_pcm_hw_params_set*
 *  function, all of the other ranges will be updated to exclude as many
 *  impossible configurations as possible. Attempting to set a parameter
 *  outside of its acceptable range will result in the function failing
 *  and an error code being returned.
 */
typedef struct _snd_pcm_hw_params snd_pcm_hw_params_t;

/** PCM software configuration container */
typedef struct _snd_pcm_sw_params snd_pcm_sw_params_t;
/** PCM status container */
 typedef struct _snd_pcm_status snd_pcm_status_t;
/** PCM access types mask */
typedef struct _snd_pcm_access_mask snd_pcm_access_mask_t;
/** PCM formats mask */
typedef struct _snd_pcm_format_mask snd_pcm_format_mask_t;
/** PCM subformats mask */
typedef struct _snd_pcm_subformat_mask snd_pcm_subformat_mask_t;

/** PCM class */
typedef enum _snd_pcm_class {
	/** standard device */

	SND_PCM_CLASS_GENERIC = 0,
	/** multichannel device */
	SND_PCM_CLASS_MULTI,
	/** software modem device */
	SND_PCM_CLASS_MODEM,
	/** digitizer device */
	SND_PCM_CLASS_DIGITIZER,
	SND_PCM_CLASS_LAST = SND_PCM_CLASS_DIGITIZER
} snd_pcm_class_t;

/** PCM subclass */
typedef enum _snd_pcm_subclass {
	/** subdevices are mixed together */
	SND_PCM_SUBCLASS_GENERIC_MIX = 0,
	/** multichannel subdevices are mixed together */
	SND_PCM_SUBCLASS_MULTI_MIX,
	SND_PCM_SUBCLASS_LAST = SND_PCM_SUBCLASS_MULTI_MIX
} snd_pcm_subclass_t;

/** PCM stream (direction) */
typedef enum _snd_pcm_stream {
	/** Playback stream */
	SND_PCM_STREAM_PLAYBACK = 0,
	/** Capture stream */
	SND_PCM_STREAM_CAPTURE,
	SND_PCM_STREAM_LAST = SND_PCM_STREAM_CAPTURE
} snd_pcm_stream_t;

/** PCM access type */
typedef enum _snd_pcm_access {
	/** mmap access with simple interleaved channels */
	SND_PCM_ACCESS_MMAP_INTERLEAVED = 0,
	/** mmap access with simple non interleaved channels */
	SND_PCM_ACCESS_MMAP_NONINTERLEAVED,
	/** mmap access with complex placement */
	SND_PCM_ACCESS_MMAP_COMPLEX,
	/** snd_pcm_readi/snd_pcm_writei access */
	SND_PCM_ACCESS_RW_INTERLEAVED,
	/** snd_pcm_readn/snd_pcm_writen access */
	SND_PCM_ACCESS_RW_NONINTERLEAVED,
	SND_PCM_ACCESS_LAST = SND_PCM_ACCESS_RW_NONINTERLEAVED
} snd_pcm_access_t;

/** PCM sample format */
typedef enum _snd_pcm_format {
	/** Unknown */
	SND_PCM_FORMAT_UNKNOWN = -1,
	/** Signed 8 bit */
	SND_PCM_FORMAT_S8 = 0,
	/** Unsigned 8 bit */
	SND_PCM_FORMAT_U8,
	/** Signed 16 bit Little Endian