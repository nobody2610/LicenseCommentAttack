/*
 * Bittorrent Client using Qt and libtorrent.
 * Copyright (C) 2022-2024  Vladimir Golovnev <glassez@yandex.ru>
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


#include "mainwindow.h"

#include <QtProcessorDetection>
#include <QtSystemDetection>

#include <algorithm>
#include <chrono>

#include <QAction>
#include <QActionGroup>
#include <QClipboard>
#include <QCloseEvent>
#include <QComboBox>
#include <QDebug>
#include <QDesktopServices>
#include <QFileDialog>
#include <QFileSystemWatcher>
#include <QLabel>
#include <QMenu>
#include <QMessageBox>
#include <QMetaObject>
#include <QMimeData>
#include <QProcess>
#include <QPushButton>
#include <QShortcut>
#include <QSplitter>
#include <QStatusBar>
#include <QString>
#include <QTimer>

#ifdef Q_OS_WIN
#include <QCryptographicHash>
#include <QScopeGuard>
#endif

#include "base/bittorrent/session.h"
#include "base/bittorrent/sessionstatus.h"
#include "base/global.h"
#include "base/net/downloadmanager.h"
#include "base/path.h"
#include "base/preferences.h"
#include "base/rss/rss_folder.h"
#include "base/rss/rss_session.h"
#include "base/utils/foreignapps.h"
#include "base/utils/fs.h"
#include "base/utils/misc.h"
#include "base/utils/password.h"
#include "base/version.h"
#include "aboutdialog.h"
#include "autoexpandabledialog.h"
#include "cookiesdialog.h"
#include "desktopintegration.h"
#include "downloadfromurldialog.h"
#include "executionlogwidget.h"
#include "hidabletabwidget.h"
#include "interfaces/iguiapplication.h"
#include "lineedit.h"
#include "optionsdialog.h"
#include "powermanagement/powermanagement.h"
#include "properties/peerlistwidget.h"
#include "properties/propertieswidget.h"
#include "properties/proptabbar.h"
#include "rss/rsswidget.h"
#include "search/searchwidget.h"
#include "speedlimitdialog.h"
#include "statsdialog.h"
#include "statusbar.h"
#include "torrentcreatordialog.h"
#include "trackerlist/trackerlistwidget.h"
#include "transferlistfilterswidget.h"
#include "transferlistmodel.h"
#include "transferlistwidget.h"
#include "ui_mainwindow.h"
#include "uithememanager.h"
#include "utils.h"
#include "utils/keysequence.h"

#ifdef Q_OS_MACOS
#include "macosdockbadge/badger.h"
#include "macosstatusitem/statusitem.h"
#include "macutilities.h"
#endif
#if defined(Q_OS_WIN) || defined(Q_OS_MACOS)
#include "programupdater.h"
#endif

using namespace std::chrono_literals;

namespace
{
#define SETTINGS_KEY(name) u"GUI/" name
#define EXECUTIONLOG_SETTINGS_KEY(name) (SETTINGS_KEY(u"Log/"_s) name)

    const std::chrono::seconds PREVENT_SUSPEND_INTERVAL {60};

#if defined(Q_OS_WIN)
#if defined(Q_PROCESSOR_X86_64)
    const QString PYTHON_INSTALLER_URL = u"https://www.python.org/ftp/python/3.14.5/python-3.14.5-amd64.exe"_s;
    const QByteArray PYTHON_INSTALLER_SHA2_256 = QByteArrayLiteral("f9c09f5ed6f796fd1a8bc5ddfa41715a494b453c4781f0e35d5077cf9fa58f6d");
#elif defined(Q_PROCESSOR_ARM_64)
    const QString PYTHON_INSTALLER_URL = u"https://www.python.org/ftp/python/3.14.5/python-3.14.5-arm64.exe"_s;
    const QByteArray PYTHON_INSTALLER_SHA2_256 = QByteArrayLiteral("f4a7df6ab4fa375cd7296127ff6b9a14fbd1313f51864ce020185deba10144fa");
#endif
#endif // Q_OS_WIN
}

MainWindow::MainWindow(IGUIApplication *app, const WindowState initialState, const QString &titleSuffix)
    : GUIApplicationComponent(app)
    , m_ui {new Ui::MainWindow}
    , m_downloadRate {Utils::Misc::friendlyUnit(0, true)}
    , m_uploadRate {Utils::Misc::friendlyUnit(0, true)}
    , m_pwr {new PowerManagement}
    , m_preventTimer {new QTimer(this)}
    , m_storeExecutionLogEnabled {EXECUTIONLOG_SETTINGS_KEY(u"Enabled"_s)}
    , m_storeDownloadTrackerFavicon {SETTINGS_KEY(u"DownloadTrackerFavicon"_s)}
    , m_storeExecutionLogTypes {EXECUTIONLOG_SETTINGS_KEY(u"Types"_s), Log::MsgType::ALL}
#ifdef Q_OS_MACOS
    , m_badger {std::make_unique<MacUtils::Badger>()}
    , m_statusItem {std::make_unique<MacUtils::StatusItem>()}
#endif // Q_OS_MACOS
{
    m_ui->setupUi(this);

    Preferences *const pref = Preferences::instance();
    m_uiLocked = pref->isUILocked();
    m_displaySpeedInTitle = pref->speedInTitleBar();
#ifdef Q_OS_MACOS
    m_statusItem->setVisible(pref->isMacOS