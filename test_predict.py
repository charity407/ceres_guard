import brain

# sanity check – prediction should include a non-null risk_level and threat_type
res = brain.predict_grain_risk('Maize', 25, 60, 800)
print(res)
assert res.get('risk_level') in ('Safe', 'Warning', 'Critical'), \
    "risk_level should be one of the expected strings"
assert res.get('threat_type'), "threat_type must not be empty"
# legacy compatibility
assert res.get('level') == res['risk_level'], "legacy 'level' key must match risk_level"
assert res.get('threat') == res['threat_type'], "legacy 'threat' key must match threat_type"
