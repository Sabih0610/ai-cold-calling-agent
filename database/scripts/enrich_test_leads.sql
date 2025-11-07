BEGIN;

WITH enrich AS (
  SELECT *
  FROM (VALUES
    -- Austin demo leads
    (3212760, 'Medicine & Healthcare',        'Medical Clinics',                       '$1 - 10M',  '10 - 50'),
    (3212763, 'Tourism & Hospitality',        'Tour Operators',                        '$1 - 10M',  '10 - 50'),
    (3212764, 'Retail',                       'General Merchandise',                   '$1 - 10M',  '25 - 100'),
    (3212765, 'E-Commerce',                   'Online Retail',                         '$1 - 10M',  '10 - 50'),
    (3212766, 'Pharmaceuticals',              'Biotech & Pharma',                      '$10 - 50M', '100 - 250'),
    (3212767, 'Internet Service',             'Internet Service Providers (ISP)',      '$1 - 10M',  '25 - 100'),
    (3212768, 'Clothing & Fashion',           'Apparel & Accessories',                 '$1 - 10M',  '10 - 50'),
    (3212769, 'Tobacco',                      'Manufactured Tobacco',                  '$10 - 50M', '100 - 250'),
    (3212770, 'Utilities',                    'Water & Wastewater',                    '$10 - 50M', '100 - 250'),
    (3212771, 'Retail',                       'Specialty Retail',                      '$1 - 10M',  '25 - 100'),
    (3212772, 'E-Commerce',                   'Direct-to-Consumer Brand',              '$1 - 10M',  '10 - 50'),

    -- Alpha campaign
    (9100000,  'Pharmaceuticals',             'R&D / Clinical',                        '$10 - 50M', '100 - 250'),
    (9100001,  'Internet Service',            'Cloud / Hosting / ISP',                 '$1 - 10M',  '25 - 100'),
    (9100002,  'E-Commerce',                  'Marketplace Platform',                  '$1 - 10M',  '10 - 50'),

    -- Beta campaign
    (9200000,  'Retail',                      'Automotive Accessories Retail',         '$1 - 10M',  '25 - 100'),
    (9200001,  'Utilities',                   'Water Treatment Services',              '$1 - 10M',  '25 - 100'),
    (9200002,  'Medicine & Healthcare',       'Healthcare IT / Systems',               '$1 - 10M',  '25 - 100'),

    -- Gamma campaign
    (9300000,  'Medicine & Healthcare',       'Medical Practice Management',           '$1 - 10M',  '25 - 100'),
    (9300001,  'Pharmaceuticals',             'Manufacturing / QA',                    '$10 - 50M', '100 - 250'),
    (9300002,  'E-Commerce',                  'B2B Software Subscriptions',            '$1 - 10M',  '25 - 100')
  ) AS t(id, industry, sub_industry, revenue, employees)
)
UPDATE leads l
SET source_row = jsonb_set(
                  jsonb_set(
                  jsonb_set(
                  jsonb_set(
                    COALESCE(l.source_row, '{}'::jsonb),
                    '{Industry}',         to_jsonb(e.industry),      true),
                    '{Sub Industry}',     to_jsonb(e.sub_industry),  true),
                    '{Revenue}',          to_jsonb(e.revenue),       true),
                    '{Employee}',         to_jsonb(e.employees),     true
                ) || jsonb_build_object('Number of Employees', e.employees)
FROM enrich e
WHERE l.id = e.id;

COMMIT;
