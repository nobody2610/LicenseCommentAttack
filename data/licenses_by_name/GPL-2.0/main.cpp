/*
 * Bittorrent Client using Qt and libtorrent.
 * Copyright (C) 2014-2026  Vladimir Golovnev <glassez@yandex.ru>
 * Copyright (C) 2006  Christophe Dumez <chris@qbittorrent.org>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 *
 * In addition, as a special exception, the copyright holders give permission to
 * link this program with the OpenSSL project's "OpenSSL" library (or with
 * modified versions of it that use the same license as the "OpenSSL" library),
 * and distribute the linked executables. You must obey the GNU General Public
 * License in all respects for all of the code used other than "OpenSSL".  If you
 * modify file(s), you may extend this exception to your version of the file(s),
 * but you are not obligated to do so. If you do not wish to do so, delete this
 * exception statement from your version.
 */


#include <QtSystemDetection>

#include <chrono>
#include <cstdlib>
#include <memory>

#ifdef Q_OS_UNIX
#include <sys/resource.h>
#endif

#ifndef Q_OS_WIN
#ifndef Q_OS_HAIKU
#include <unistd.h>
#endif // Q_OS_HAIKU
#elif defined DISABLE_GUI
#include <io.h>
#endif

#include <QCoreApplication>
#include <QString>
#include <QThread>

#ifndef DISABLE_GUI
// GUI-only includes
#include <QFont>
#include <QMessageBox>
#include <QPainter>
#include <QPen>
#include <QSplashScreen>
#include <QTimer>

#ifdef QBT_STATIC_QT
#include <QtPlugin>
Q_IMPORT_PLUGIN(QICOPlugin)
#endif // QBT_STATIC_QT

#else // DISABLE_GUI
#include <cstdio>
#endif // DISABLE_GUI

#include "base/preferences.h"
#include "base/profile.h"
#include "base/settingvalue.h"
#include "base/version.h"
#include "application.h"
#include "cmdoptions.h"
#include "legalnotice.h"
#include "signalhandler.h"

#ifndef DISABLE_GUI
#include "gui/utils.h"
#else
#ifndef Q_OS_WIN
#include "base/logger.h"
#endif
#endif

using namespace std::chrono_literals;
using namespace Qt::Literals::StringLiterals;

namespace
{
    void displayBadArgMessage(const QString &message)
    {
        const QString help = QCoreApplication::translate("Main", "Run application with -h option to read about command line parameters.");
#if defined(Q_OS_WIN) && !defined(DISABLE_GUI)
        QMessageBox msgBox(QMessageBox::Critical, QCoreApplication::translate("Main", "Bad command line"),
                           (message + u'\n' + help), QMessageBox::Ok);
        msgBox.show(); // Need to be shown or to moveToCenter does not work
        msgBox.move(Utils::Gui::screenCenter(&msgBox));
        msgBox.exec();
#else
        const QString errMsg = QCoreApplication::translate("Main", "Bad command line: ") + u'\n'
            + message + u'\n'
            + help + u'\n';
        fprintf(stderr, "%s", qUtf8Printable(errMsg));
#endif
    }

    void displayErrorMessage(const QString &message)
    {
#ifndef DISABLE_GUI
        if (QApplication::instance())
        {
            QMessageBox msgBox;
            msgBox.setIcon(QMessageBox::Critical);
            msgBox.setText(QCoreApplication::translate("Main", "An unrecoverable error occurred."));
            msgBox.setInformativeText(message);
            msgBox.show(); // Need to be shown or to moveToCenter does not work
            msgBox.move(Utils::Gui::screenCenter(&msgBox));
            msgBox.exec();
        }
        else
        {
            const QString errMsg = QCoreApplication::translate("Main", "qBittorrent has encountered an unrecoverable error.") + u'\n' + message + u'\n';
            fprintf(stderr, "%s", qUtf8Printable(errMsg));
        }
#else
        const QString errMsg = QCoreApplication::translate("Main", "qBittorrent has encountered an unrecoverable error.") + u'\n' + message + u'\n';
        fprintf(stderr, "%s", qUtf8Printable(errMsg));
#endif
    }

#if !defined(Q_OS_WIN) || defined(DISABLE_GUI)
    void displayVersion()
    {
        printf("%s %s\n", qUtf8Printable(qApp->applicationName()), QBT_VERSION);
    }
#endif

#ifndef DISABLE_GUI
    void showSplashScreen()
    {
        QPixmap splashImg(u":/icons/splash.png"_s);
        QPainter painter(&splashImg);
        const auto version = QStringLiteral(QBT_VERSION);
        painter.setPen(QPen(Qt::white));
        painter.setFont(QFont(u"Arial"_s, 22, QFont::Black));
        painter.drawText(224 - painter.fontMetrics().horizontalAdvance(version), 270, version);
        QSplashScreen *splash = new QSplashScreen(splashImg);
        splash->show();
        QTimer::singleShot(1500ms, Qt::CoarseTimer, splash, &QObject::deleteLater);
        qApp->processEvents();
    }
#endif  // DISABLE_GUI

#ifdef Q_OS_UNIX
    void adjustFileDescriptorLimit()
    {
        rlimit limit {};

        if (getrlimit(RLIMIT_NOFILE, &limit) != 0)
            return;

        limit.rlim_cur = limit.rlim_max;
        setrlimit(RLIMIT_NOFILE, &limit);
    }

    void adjustLocale()
    {
        // specify the default loca