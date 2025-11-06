BEGIN;

----------------------------------------------------------------
-- Campaign 1: Demo Campaign Alpha
----------------------------------------------------------------
WITH new_campaign AS (
    INSERT INTO campaigns (name, slug, notes)
    VALUES ('Demo Campaign Alpha', 'demo-campaign-alpha', 'Dummy leads for switching tests – set Alpha')
    RETURNING id
),
lead_rows AS (
    INSERT INTO leads (
        id, company, contact, email, phone_raw, phone_e164, website,
        address, city, state, postal_code, country_code, tz_name,
        source_first_seen, last_seen, source_file, source_row
    )
    VALUES
    (
        9100000,
        'AlphaCo One',
        'Morgan Avery',
        'info+alpha1@example.test',
        '+15005558001',
        '+15005558001',
        'https://alphaco1.example.test',
        '10 Congress Ave',
        'Austin',
        'TX',
        '78701',
        'US',
        'America/Chicago',
        now(),
        now(),
        'alpha_seed.csv',
        jsonb_build_object(
            'Id', 9100000,
            'City', 'Austin',
            'E164', '+15005558001',
            'Phone', '+15005558001',
            'State', 'TX',
            'Title', null,
            'Country', 'USA',
            'Revenue', 'N/A',
            'TZ Name', 'America/Chicago',
            'ZIP Code', '78701',
            'Phone Type', 'fixed line',
            'Source File', 'alpha_seed.csv',
            'Web Address', 'https://alphaco1.example.test',
            'Primary City', 'Austin',
            'Business Name', 'AlphaCo One',
            'Generic Email', 'info+alpha1@example.test',
            'Primary State', 'TX',
            'Contact Person', 'Morgan Avery',
            'Street Address', '10 Congress Ave',
            'Number of Employees', 'N/A'
        )
    ),
    (
        9100001,
        'AlphaCo Two',
        'Reese Bennett',
        'info+alpha2@example.test',
        '+15005558002',
        '+15005558002',
        'https://alphaco2.example.test',
        '20 Congress Ave',
        'Austin',
        'TX',
        '78702',
        'US',
        'America/Chicago',
        now(),
        now(),
        'alpha_seed.csv',
        jsonb_build_object(
            'Id', 9100001,
            'City', 'Austin',
            'E164', '+15005558002',
            'Phone', '+15005558002',
            'State', 'TX',
            'Title', null,
            'Country', 'USA',
            'Revenue', 'N/A',
            'TZ Name', 'America/Chicago',
            'ZIP Code', '78702',
            'Phone Type', 'fixed line',
            'Source File', 'alpha_seed.csv',
            'Web Address', 'https://alphaco2.example.test',
            'Primary City', 'Austin',
            'Business Name', 'AlphaCo Two',
            'Generic Email', 'info+alpha2@example.test',
            'Primary State', 'TX',
            'Contact Person', 'Reese Bennett',
            'Street Address', '20 Congress Ave',
            'Number of Employees', 'N/A'
        )
    ),
    (
        9100002,
        'AlphaCo Three',
        'Skyler Daniels',
        'info+alpha3@example.test',
        '+15005558003',
        '+15005558003',
        'https://alphaco3.example.test',
        '30 Congress Ave',
        'Austin',
        'TX',
        '78703',
        'US',
        'America/Chicago',
        now(),
        now(),
        'alpha_seed.csv',
        jsonb_build_object(
            'Id', 9100002,
            'City', 'Austin',
            'E164', '+15005558003',
            'Phone', '+15005558003',
            'State', 'TX',
            'Title', null,
            'Country', 'USA',
            'Revenue', 'N/A',
            'TZ Name', 'America/Chicago',
            'ZIP Code', '78703',
            'Phone Type', 'fixed line',
            'Source File', 'alpha_seed.csv',
            'Web Address', 'https://alphaco3.example.test',
            'Primary City', 'Austin',
            'Business Name', 'AlphaCo Three',
            'Generic Email', 'info+alpha3@example.test',
            'Primary State', 'TX',
            'Contact Person', 'Skyler Daniels',
            'Street Address', '30 Congress Ave',
            'Number of Employees', 'N/A'
        )
    )
    RETURNING id, city, state, source_file
)
INSERT INTO campaign_leads (campaign_id, lead_id, source_city, source_state, source_file)
SELECT new_campaign.id, l.id, l.city, l.state, l.source_file
FROM new_campaign CROSS JOIN lead_rows l;

----------------------------------------------------------------
-- Campaign 2: Demo Campaign Beta
----------------------------------------------------------------
WITH new_campaign AS (
    INSERT INTO campaigns (name, slug, notes)
    VALUES ('Demo Campaign Beta', 'demo-campaign-beta', 'Dummy leads for switching tests – set Beta')
    RETURNING id
),
lead_rows AS (
    INSERT INTO leads (
        id, company, contact, email, phone_raw, phone_e164, website,
        address, city, state, postal_code, country_code, tz_name,
        source_first_seen, last_seen, source_file, source_row
    )
    VALUES
    (
        9200000,
        'Beta Builders',
        'Peyton Ellis',
        'info+beta1@example.test',
        '+15005558101',
        '+15005558101',
        'https://betabuilders1.example.test',
        '410 Elm St',
        'Dallas',
        'TX',
        '75201',
        'US',
        'America/Chicago',
        now(),
        now(),
        'beta_seed.csv',
        jsonb_build_object(
            'Id', 9200000,
            'City', 'Dallas',
            'E164', '+15005558101',
            'Phone', '+15005558101',
            'State', 'TX',
            'Title', null,
            'Country', 'USA',
            'Revenue', 'N/A',
            'TZ Name', 'America/Chicago',
            'ZIP Code', '75201',
            'Phone Type', 'fixed line',
            'Source File', 'beta_seed.csv',
            'Web Address', 'https://betabuilders1.example.test',
            'Primary City', 'Dallas',
            'Business Name', 'Beta Builders',
            'Generic Email', 'info+beta1@example.test',
            'Primary State', 'TX',
            'Contact Person', 'Peyton Ellis',
            'Street Address', '410 Elm St',
            'Number of Employees', 'N/A'
        )
    ),
    (
        9200001,
        'Beta Labs',
        'Casey Flores',
        'info+beta2@example.test',
        '+15005558102',
        '+15005558102',
        'https://betalabs.example.test',
        '920 Commerce St',
        'Dallas',
        'TX',
        '75202',
        'US',
        'America/Chicago',
        now(),
        now(),
        'beta_seed.csv',
        jsonb_build_object(
            'Id', 9200001,
            'City', 'Dallas',
            'E164', '+15005558102',
            'Phone', '+15005558102',
            'State', 'TX',
            'Title', null,
            'Country', 'USA',
            'Revenue', 'N/A',
            'TZ Name', 'America/Chicago',
            'ZIP Code', '75202',
            'Phone Type', 'fixed line',
            'Source File', 'beta_seed.csv',
            'Web Address', 'https://betalabs.example.test',
            'Primary City', 'Dallas',
            'Business Name', 'Beta Labs',
            'Generic Email', 'info+beta2@example.test',
            'Primary State', 'TX',
            'Contact Person', 'Casey Flores',
            'Street Address', '920 Commerce St',
            'Number of Employees', 'N/A'
        )
    ),
    (
        9200002,
        'Beta Systems',
        'Jamie Ortiz',
        'info+beta3@example.test',
        '+15005558103',
        '+15005558103',
        'https://betasystems.example.test',
        '1330 Griffin St',
        'Dallas',
        'TX',
        '75203',
        'US',
        'America/Chicago',
        now(),
        now(),
        'beta_seed.csv',
        jsonb_build_object(
            'Id', 9200002,
            'City', 'Dallas',
            'E164', '+15005558103',
            'Phone', '+15005558103',
            'State', 'TX',
            'Title', null,
            'Country', 'USA',
            'Revenue', 'N/A',
            'TZ Name', 'America/Chicago',
            'ZIP Code', '75203',
            'Phone Type', 'fixed line',
            'Source File', 'beta_seed.csv',
            'Web Address', 'https://betasystems.example.test',
            'Primary City', 'Dallas',
            'Business Name', 'Beta Systems',
            'Generic Email', 'info+beta3@example.test',
            'Primary State', 'TX',
            'Contact Person', 'Jamie Ortiz',
            'Street Address', '1330 Griffin St',
            'Number of Employees', 'N/A'
        )
    )
    RETURNING id, city, state, source_file
)
INSERT INTO campaign_leads (campaign_id, lead_id, source_city, source_state, source_file)
SELECT new_campaign.id, l.id, l.city, l.state, l.source_file
FROM new_campaign CROSS JOIN lead_rows l;

----------------------------------------------------------------
-- Campaign 3: Demo Campaign Gamma
----------------------------------------------------------------
WITH new_campaign AS (
    INSERT INTO campaigns (name, slug, notes)
    VALUES ('Demo Campaign Gamma', 'demo-campaign-gamma', 'Dummy leads for switching tests – set Gamma')
    RETURNING id
),
lead_rows AS (
    INSERT INTO leads (
        id, company, contact, email, phone_raw, phone_e164, website,
        address, city, state, postal_code, country_code, tz_name,
        source_first_seen, last_seen, source_file, source_row
    )
    VALUES
    (
        9300000,
        'Gamma Group',
        'Hayden Price',
        'info+gamma1@example.test',
        '+15005558201',
        '+15005558201',
        'https://gammagroup.example.test',
        '1010 Prairie St',
        'Houston',
        'TX',
        '77002',
        'US',
        'America/Chicago',
        now(),
        now(),
        'gamma_seed.csv',
        jsonb_build_object(
            'Id', 9300000,
            'City', 'Houston',
            'E164', '+15005558201',
            'Phone', '+15005558201',
            'State', 'TX',
            'Title', null,
            'Country', 'USA',
            'Revenue', 'N/A',
            'TZ Name', 'America/Chicago',
            'ZIP Code', '77002',
            'Phone Type', 'fixed line',
            'Source File', 'gamma_seed.csv',
            'Web Address', 'https://gammagroup.example.test',
            'Primary City', 'Houston',
            'Business Name', 'Gamma Group',
            'Generic Email', 'info+gamma1@example.test',
            'Primary State', 'TX',
            'Contact Person', 'Hayden Price',
            'Street Address', '1010 Prairie St',
            'Number of Employees', 'N/A'
        )
    ),
    (
        9300001,
        'Gamma Labs',
        'Rowan West',
        'info+gamma2@example.test',
        '+15005558202',
        '+15005558202',
        'https://gammalabs.example.test',
        '2250 Lamar St',
        'Houston',
        'TX',
        '77003',
        'US',
        'America/Chicago',
        now(),
        now(),
        'gamma_seed.csv',
        jsonb_build_object(
            'Id', 9300001,
            'City', 'Houston',
            'E164', '+15005558202',
            'Phone', '+15005558202',
            'State', 'TX',
            'Title', null,
            'Country', 'USA',
            'Revenue', 'N/A',
            'TZ Name', 'America/Chicago',
            'ZIP Code', '77003',
            'Phone Type', 'fixed line',
            'Source File', 'gamma_seed.csv',
            'Web Address', 'https://gammalabs.example.test',
            'Primary City', 'Houston',
            'Business Name', 'Gamma Labs',
            'Generic Email', 'info+gamma2@example.test',
            'Primary State', 'TX',
            'Contact Person', 'Rowan West',
            'Street Address', '2250 Lamar St',
            'Number of Employees', 'N/A'
        )
    ),
    (
        9300002,
        'Gamma Solutions',
        'Taylor Young',
        'info+gamma3@example.test',
        '+15005558203',
        '+15005558203',
        'https://gammasolutions.example.test',
        '3310 Milam St',
        'Houston',
        'TX',
        '77006',
        'US',
        'America/Chicago',
        now(),
        now(),
        'gamma_seed.csv',
        jsonb_build_object(
            'Id', 9300002,
            'City', 'Houston',
            'E164', '+15005558203',
            'Phone', '+15005558203',
            'State', 'TX',
            'Title', null,
            'Country', 'USA',
            'Revenue', 'N/A',
            'TZ Name', 'America/Chicago',
            'ZIP Code', '77006',
            'Phone Type', 'fixed line',
            'Source File', 'gamma_seed.csv',
            'Web Address', 'https://gammasolutions.example.test',
            'Primary City', 'Houston',
            'Business Name', 'Gamma Solutions',
            'Generic Email', 'info+gamma3@example.test',
            'Primary State', 'TX',
            'Contact Person', 'Taylor Young',
            'Street Address', '3310 Milam St',
            'Number of Employees', 'N/A'
        )
    )
    RETURNING id, city, state, source_file
)
INSERT INTO campaign_leads (campaign_id, lead_id, source_city, source_state, source_file)
SELECT new_campaign.id, l.id, l.city, l.state, l.source_file
FROM new_campaign CROSS JOIN lead_rows l;

COMMIT;
