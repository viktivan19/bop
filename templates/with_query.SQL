WITH tx AS (select correspondent_schedule

                 , DATE(convert_timezone('Europe/London'::text,
                                         eve.event_date::timestamp without time zone)) AS paid_date
                 , CASE
                       WHEN dpm.payout_method_code = 'ATP' THEN recv_cur.currency_code
                       ELSE cor_cur.currency_code END                                  AS settlement_currency
                 , CASE
                       WHEN eve.event_type_id = 14 OR eve.event_type_id = 4 THEN 1
                       WHEN eve.event_type_id = 23 THEN -1 END                         AS multiplier
                 , (trx.correspondent_amount_corr_curr +
                    trx.correspondent_commission_corr_curr) *
                   multiplier                                                          AS recv_amount_local
                 , CASE
                       WHEN dpm.payout_method_code = 'ATP'
                           THEN recv_amount_local * (1 / fx_rate_usd_to_recv)
                       ELSE recv_amount_local * (1 / fx_rate_usd_to_corr_creation) END AS recv_amount_usd

            FROM _dwh_pii_plus_fraud.f_transaction_event eve
                     LEFT JOIN _dwh_pii_plus_fraud.f_transaction trx ON eve.transaction_id = trx.transaction_id


            WHERE eve.event_type_id = CASE
                                          WHEN correspondent_group_name = 'WorldPay'
                                              THEN 14 OR 23
                                          ELSE 4 OR 23 END
              and DATE(convert_timezone('Europe/London'::text,
                                        eve.event_date::timestamp without time zone)) BETWEEN DATE(DATEADD(DAY, -1035, GETDATE())) AND DATE(GETDATE())
)

SELECT correspondent_schedule,
       settlement_currency AS currency_code,

from tx
GROUP BY 1, 2
ORDER BY 1, 2