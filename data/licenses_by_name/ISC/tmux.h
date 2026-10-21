/* $OpenBSD$ */
/*
 * Copyright (c) 2007 Nicholas Marriott <nicholas.marriott@gmail.com>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF MIND, USE, DATA OR PROFITS, WHETHER
 * IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
 * OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef TMUX_H
#define TMUX_H

#include <sys/time.h>
#include <sys/uio.h>

#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <termios.h>
#include <wchar.h>

#ifdef HAVE_UTEMPTER
#include <utempter.h>
#endif

#include "compat.h"
#include "tmux-protocol.h"
#include "xmalloc.h"

extern char   **environ;

struct args;
struct args_command_state;
struct client;
struct cmd;
struct cmd_find_state;
struct cmdq_item;
struct cmdq_list;
struct cmdq_state;
struct cmds;
struct control_state;
struct environ;
struct format_job_tree;
struct format_tree;
struct hyperlinks_uri;
struct hyperlinks;
struct input_ctx;
struct input_request;
struct input_requests;
struct job;
struct menu_data;
struct mode_tree_data;
struct mouse_event;
struct options;
struct options_array_item;
struct options_entry;
struct redraw_scene;
struct redraw_span;
struct screen_write_citem;
struct screen_write_cline;
struct screen_write_ctx;
struct session;

#ifdef ENABLE_SIXEL
struct sixel_image;
#endif

struct tty_ctx;
struct tty_code;
struct tty_key;
struct tmuxpeer;
struct tmuxproc;
struct winlink;

/* Default configuration files and socket paths. */
#ifndef TMUX_CONF
#define TMUX_CONF "/etc/tmux.conf:~/.tmux.conf"
#endif
#ifndef TMUX_SOCK
#define TMUX_SOCK "$TMUX_TMPDIR:" _PATH_TMP
#endif
#ifndef TMUX_SOCK_PERM
#define TMUX_SOCK_PERM (7 /* o+rwx */)
#endif
#ifndef TMUX_TERM
#define TMUX_TERM "screen"
#endif
#ifndef TMUX_LOCK_CMD
#define TMUX_LOCK_CMD "lock -np"
#endif

/* Forbidden characters in names. */
#define WINDOW_NAME_FORBID ":."
#define WINDOW_NAME_FORBID_EXT ":.#"
#define SESSION_NAME_FORBID ":."
#define SESSION_NAME_FORBID_EXT ":.#"

/* Minimum and maximum layout cell size, NOT including border lines. */
#define PANE_MINIMUM 1
#define PANE_MAXIMUM 10000

/* Minimum and maximum window size. */
#define WINDOW_MINIMUM PANE_MINIMUM
#define WINDOW_MAXIMUM 10000

/* Automatic name refresh interval, in microseconds. Must be < 1 second. */
#define NAME_INTERVAL 500000

/* Default pixel cell sizes. */
#define DEFAULT_XPIXEL 16
#define DEFAULT_YPIXEL 32

/* Attribute to make GCC check printf-like arguments. */
#define printflike(a, b) __attribute__ ((format (printf, a, b)))

/* Number of items in array. */
#ifndef nitems
#define nitems(_a) (sizeof((_a)) / sizeof((_a)[0]))
#endif

/* Alert option values. */
#define ALERT_NONE 0
#define ALERT_ANY 1
#define ALERT_CURRENT 2
#define ALERT_OTHER 3

/* Visual option values. */
#define VISUAL_OFF 0
#define VISUAL_ON 1
#define VISUAL_BOTH 2

/* Key modifier bits. */
#define KEYC_META            0x00100000000000ULL
#define KEYC_CTRL            0x00200000000000ULL
#define KEYC_SHIFT           0x00400000000000ULL

/* Key flag bits. */
#define KEYC_LITERAL	     0x01000000000000ULL
#define KEYC_KEYPAD	     0x02000000000000ULL
#define KEYC_CURSOR	     0x04000000000000ULL
#define KEYC_IMPLIED_META    0x08000000000000ULL
#define KEYC_BUILD_MODIFIERS 0x10000000000000ULL
#define KEYC_VI		     0x20000000000000ULL
#define KEYC_SENT	     0x40000000000000ULL

/* Masks for key bits. */
#define KEYC_MASK_TYPE       0x0000ff00000000ULL
#define KEYC_MASK_MODIFIERS  0x00ff0000000000ULL
#define KEYC_MASK_FLAGS      0xff000000000000ULL