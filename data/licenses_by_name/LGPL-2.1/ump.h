/* SPDX-License-Identifier: LGPL-2.1+ */
/**
 * \file include/ump.h
 * \brief API library for ALSA rawmidi/UMP interface
 *
 * API library for ALSA rawmidi/UMP interface
 */

#ifndef __ALSA_UMP_H
#define __ALSA_UMP_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  \defgroup RawMidi RawMidi Interface
 *  The RawMidi Interface. See \ref rawmidi page for more details.
 *  \{
 */

/** UMP (Endpoint) RawMIDI device */
typedef struct _snd_ump snd_ump_t;
/** UMP Endpoint information container */
typedef struct snd_ump_endpoint_info snd_ump_endpoint_info_t;
/** UMP Block information container */
typedef struct snd_ump_block_info snd_ump_block_info_t;

int snd_ump_open(snd_ump_t **inputp, snd_ump_t **outputp, const char *name, int mode);
int snd_ump_close(snd_ump_t *ump);
snd_rawmidi_t *snd_ump_rawmidi(snd_ump_t *ump);
const char *snd_ump_name(snd_ump_t *ump);
int snd_ump_poll_descriptors_count(snd_ump_t *ump);
int snd_ump_poll_descriptors(snd_ump_t *ump, struct pollfd *pfds, unsigned int space);
int snd_ump_poll_descriptors_revents(snd_ump_t *ump, struct pollfd *pfds, unsigned int nfds, unsigned short *revents);
int snd_ump_nonblock(snd_ump_t *ump, int nonblock);
int snd_ump_rawmidi_info(snd_ump_t *ump, snd_rawmidi_info_t *info);
int snd_ump_rawmidi_params(snd_ump_t *ump, snd_rawmidi_params_t *params);
int snd_ump_rawmidi_params_current(snd_ump_t *ump, snd_rawmidi_params_t *params);
int snd_ump_rawmidi_status(snd_ump_t *ump, snd_rawmidi_status_t *status);
int snd_ump_drop(snd_ump_t *ump);
int snd_ump_drain(snd_ump_t *ump);
ssize_t snd_ump_write(snd_ump_t *ump, const void *buffer, size_t size);
ssize_t snd_ump_read(snd_ump_t *ump, void *buffer, size_t size);
ssize_t snd_ump_tread(snd_ump_t *ump, struct timespec *tstamp, void *buffer, size_t size);

/** Max number of UMP Groups */
#define SND_UMP_MAX_GROUPS		16
/** Max number of UMP Blocks */
#define SND_UMP_MAX_BLOCKS		32

/** UMP direction */
enum _snd_ump_direction {
	/** Input only */
	SND_UMP_DIR_INPUT =		0x01,
	/** Output only */
	SND_UMP_DIR_OUTPUT =		0x02,
	/** Bidirectional */
	SND_UMP_DIR_BIDIRECTION	=	0x03,
};

/** UMP EP holds only static blocks */
#define SND_UMP_EP_INFO_STATIC_BLOCKS		0x01

/** Bitmask for UMP EP MIDI protocols */
#define SND_UMP_EP_INFO_PROTO_MIDI_MASK		0x0300
/** Bit flag for MIDI 1.0 protocol */
#define SND_UMP_EP_INFO_PROTO_MIDI1		0x0100
/** Bit flag for MIDI 2.0 protocol */
#define SND_UMP_EP_INFO_PROTO_MIDI2		0x0200
/** Bitmask for UMP Jitter-reduction timestamp */
#define SND_UMP_EP_INFO_PROTO_JRTS_MASK		0x0003
/** Bit flag for JRTS in Transmit */
#define SND_UMP_EP_INFO_PROTO_JRTS_TX		0x0001
/** Bit flag for JRTS in Receive */
#define SND_UMP_EP_INFO_PROTO_JRTS_RX		0x0002

/** Default version passed to UMP Endpoint info */
#define SND_UMP_EP_INFO_DEFAULT_VERSION		0x0101

size_t snd_ump_endpoint_info_sizeof(void);
/** \hideinitializer
 * \brief allocate an invalid #snd_ump_endpoint_info_t using standard alloca
 * \param ptr returned pointer
 */
#define snd_ump_endpoint_info_alloca(ptr) __snd_alloca(ptr, snd_ump_endpoint_info)
int snd_ump_endpoint_info_malloc(snd_ump_endpoint_info_t **info);
void snd_ump_endpoint_info_free(snd_ump_endpoint_info_t *info);
void snd_ump_endpoint_info_clear(snd_ump_endpoint_info_t *info);
void snd_ump_endpoint_info_copy(snd_ump_endpoint_info_t *dst, const snd_ump_endpoint_info_t *src);
int snd_ump_endpoint_info_get_card(const snd_ump_endpoint_info_t *info);
int snd_ump_endpoint_info_get_device(const snd_ump_endpoint_info_t *info);
unsigned int snd_ump_endpoint_info_get_flags(const snd_ump_endpoint_info_t *info);
unsigned int snd_ump_endpoint_info_get_protocol_caps(const snd_ump_endpoint_info_t *info);
unsigned int snd_ump_endpoint_info_get_protocol(const snd_ump_endpoint_info_t *info);
unsigned int snd_ump_endpoint_info_get_num_blocks(const snd_ump_endpoint_info_t *info);
unsigned int snd_ump_endpoint_info_get_version(const snd_ump_endpoint_info_t *info);
unsigned int snd_ump_endpoint_info_get_manufacturer_id(const snd_ump_endpoint_info_t *info);
unsigned int