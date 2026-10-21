/*
 * Copyright (C) 2020 Graylog, Inc.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the Server Side Public License, version 1,
 * as published by MongoDB, Inc.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Server Side Public License for more details.
 *
 * You should have received a copy of the Server Side Public License
 * along with this program. If not, see
 * <http://www.mongodb.com/licensing/server-side-public-license>.
 */

package org.graylog2;

import com.github.joschi.jadconfig.Parameter;
import com.github.joschi.jadconfig.ValidationException;
import com.github.joschi.jadconfig.Validator;
import com.github.joschi.jadconfig.ValidatorMethod;
import com.github.joschi.jadconfig.converters.StringSetConverter;
import com.github.joschi.jadconfig.converters.TrimmedStringSetConverter;
import com.github.joschi.jadconfig.documentation.Documentation;
import com.github.joschi.jadconfig.documentation.DocumentationSection;
import com.github.joschi.jadconfig.util.Duration;
import com.github.joschi.jadconfig.util.Size;
import com.github.joschi.jadconfig.util.SizeUnit;
import com.github.joschi.jadconfig.validators.PositiveDurationValidator;
import com.github.joschi.jadconfig.validators.PositiveIntegerValidator;
import com.github.joschi.jadconfig.validators.PositiveLongValidator;
import com.github.joschi.jadconfig.validators.StringNotBlankValidator;
import com.google.common.collect.Sets;
import org.apache.commons.lang3.StringUtils;
import org.graylog.plugins.views.search.engine.suggestions.FieldValueSuggestionMode;
import org.graylog.plugins.views.search.engine.suggestions.FieldValueSuggestionModeConverter;
import org.graylog.security.certutil.CaConfiguration;
import org.graylog2.bindings.NamedBindingOverride;
import org.graylog2.cluster.leader.AutomaticLeaderElectionService;
import org.graylog2.cluster.leader.LeaderElectionMode;
import org.graylog2.cluster.leader.LeaderElectionService;
import org.graylog2.cluster.lock.MongoLockService;
import org.graylog2.configuration.DocumentationConstants;
import org.graylog2.configuration.converters.JavaDurationConverter;
import org.graylog2.configuration.validators.PositiveJavaDurationValidator;
import org.graylog2.notifications.Notification;
import org.graylog2.outputs.BatchSizeConfig;
import org.graylog2.plugin.Tools;
import org.graylog2.security.hashing.PBKDF2PasswordAlgorithm;
import org.graylog2.security.realm.RootAccountRealm;
import org.graylog2.utilities.IPSubnetConverter;
import org.graylog2.utilities.IpSubnet;
import org.joda.time.DateTimeZone;
import org.joda.time.Period;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.Optional;
import java.util.Set;

import static org.graylog2.shared.utilities.StringUtils.f;

/**
 * Helper class to hold configuration of Graylog
 */
@SuppressWarnings("FieldMayBeFinal")
@DocumentationSection(heading = "GRAYLOG CONFIGURATION FILE", description = DocumentationConstants.SERVER_DOCUMENTATION_DESCRIPTION)
public class Configuration extends CaConfiguration implements CommonNodeConfiguration {
    public static final String SAFE_CLASSES = "safe_classes";

    public static final String CONTENT_PACKS_DIR = "content_packs_dir";
    private static final String NODE_ID_FILE = "node_id_file";
    /**
     * Deprecated! Use isLeader() instead.
     */
    @Documentation(visible = false)
    @Parameter(value = "is_master")
    private boolean isMaster = true;

    @Documentation("""
            If set to "true", Graylog will periodically investigate indices to figure out which fields are used in which streams.
            It will make field list in Graylog interface show only fields used in selected streams, but can decrease system performance,
            especially on systems with great number of streams and fields.
            """)
    @Parameter(value = "stream_aware_field_types")
    private boolean streamAwareFieldTypes = false;

    /**
     * Used for initializing static leader election. You shouldn't use this for other purposes, but if you must, don't
     * use @{@link jakarta.inject.Named} injection but the getter isLeader() instead.
     **/
    @Documentation("""
            If you are running more than one instances of Graylog server you have to select one of these
            instances as leader. The leader will perform some periodical tasks that non-leaders won't perform.
            """)
    @Parameter