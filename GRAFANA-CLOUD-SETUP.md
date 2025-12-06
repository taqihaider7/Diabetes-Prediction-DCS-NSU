# Grafana Cloud Setup Guide

Complete guide to integrate your Diabetes Prediction ML app with Grafana Cloud for monitoring.

## 🎯 Why Grafana Cloud?

- ✅ **Free Tier**: 10,000 series, 50GB logs, 50GB traces
- ✅ **No Infrastructure**: No Docker/servers to manage
- ✅ **Fully Managed**: Automatic updates and scaling
- ✅ **Global**: Low latency worldwide
- ✅ **Integrated**: Loki, Prometheus, Tempo, Grafana in one place

## 📋 Prerequisites

- Email address for account creation
- Python application running
- 10 minutes of setup time

## 🚀 Step-by-Step Setup

### Step 1: Create Grafana Cloud Account

1. **Visit Grafana Cloud**
   - Go to: https://grafana.com/auth/sign-up/create-user
   
2. **Sign Up**
   - Enter your email
   - Choose a strong password
   - Verify your email

3. **Choose Plan**
   - Select **Free** tier
   - No credit card required

### Step 2: Get Your Loki Credentials

1. **Access Cloud Portal**
   - Login to: https://grafana.com/
   - Click on your organization name

2. **Navigate to Loki**
   - Go to: **Connections** → **Add new connection**
   - Search for "Loki"
   - Click **Configure**

3. **Copy Your Loki URL**
   ```
   Format: https://logs-prod-XXX.grafana.net/loki/api/v1/push
   
   Example: https://logs-prod-us-central1.grafana.net/loki/api/v1/push
   ```
   
   **Save this URL!**

### Step 3: Create API Key

1. **Generate API Key**
   - Go to: **Cloud Portal** → **Security** → **API Keys**
   - Or direct: https://grafana.com/orgs/YOUR-ORG/api-keys

2. **Create New Key**
   - Click **Add API Key**
   - **Name**: `diabetes-prediction-logs`
   - **Role**: Select **MetricsPublisher**
   - **Time to live**: Leave empty (no expiration)
   - Click **Add**

3. **Copy API Key**
   ```
   Format: glc_XXXXXXXXXXXXXXXXXXXXXXXXXXXXX
   ```
   
   **⚠️ IMPORTANT: Copy this now! You won't see it again.**

### Step 4: Get Your Instance ID (Username)

1. **Find Instance Details**
   - In Cloud Portal, go to **Grafana**
   - Click on your Grafana instance
   - Copy the **Instance ID** or **User ID**
   
   ```
   Format: 123456  (just numbers)
   ```

### Step 5: Configure Your Application

1. **Create `.env` file** in project root:
   ```bash
   # Grafana Cloud Configuration
   LOKI_ENABLED=true
   LOKI_URL=https://logs-prod-XXX.grafana.net/loki/api/v1/push
   LOKI_USERNAME=123456
   LOKI_API_KEY=glc_XXXXXXXXXXXXXXXXXXXXXXXXXXXXX
   
   # Application Settings
   APP_NAME=diabetes-prediction
   ENVIRONMENT=production
   LOG_LEVEL=INFO
   METRICS_RETENTION_HOURS=24
   TRACK_USER_SESSIONS=true
   ```

2. **Replace placeholders:**
   - `XXX` in LOKI_URL with your actual URL
   - `123456` with your instance ID
   - `glc_XXX...` with your actual API key

### Step 6: Test Connection

1. **Run the main app:**
   ```bash
   cd src
   streamlit run app.py
   ```

2. **Make a test prediction**
   - Go to http://localhost:8501
   - Navigate to "Single Prediction"
   - Fill in features and predict

3. **Check logs in Grafana Cloud:**
   - Open your Grafana instance
   - Go to **Explore**
   - Select **Loki** data source
   - Query: `{app="diabetes-prediction"}`
   - You should see your logs!

### Step 7: Run Monitoring Dashboard

1. **Start monitoring app:**
   ```bash
   cd src
   streamlit run monitoring_app.py --server.port 8502
   ```

2. **Access dashboard:**
   - Open: http://localhost:8502
   - Navigate through different pages
   - See real-time metrics

## 📊 Grafana Dashboards

### Creating Your First Dashboard

1. **In Grafana Cloud, click:**
   - **Dashboards** → **New** → **New Dashboard**

2. **Add Panel for Logs:**
   - Click **Add visualization**
   - Select **Loki** data source
   - Query: 
     ```logql
     {app="diabetes-prediction"}
     ```
   - Click **Run query**
   - Save panel

3. **Add Panel for Error Rate:**
   - Add new panel
   - Query:
     ```logql
     sum by (level) (
       rate({app="diabetes-prediction"}[5m])
     )
     ```
   - Visualization: **Time series**
   - Save

4. **Add Panel for Latency:**
   - Add new panel
   - Query:
     ```logql
     quantile_over_time(0.95, 
       {app="diabetes-prediction"} 
       | json 
       | unwrap duration_ms [5m]
     )
     ```
   - Visualization: **Gauge**
   - Save

### Useful Queries

**All application logs:**
```logql
{app="diabetes-prediction"}
```

**Only errors:**
```logql
{app="diabetes-prediction", level="error"}
```

**Successful predictions:**
```logql
{app="diabetes-prediction"} |= "Prediction successful"
```

**High latency (>100ms):**
```logql
{app="diabetes-prediction"} 
| json 
| duration_ms > 100
```

**Error rate over time:**
```logql
sum by (level) (
  rate({app="diabetes-prediction", level="error"}[5m])
)
```

**P95 latency:**
```logql
quantile_over_time(0.95, 
  {app="diabetes-prediction"} 
  | json 
  | unwrap duration_ms [5m]
)
```

**Predictions by class:**
```logql
sum by (prediction_result) (
  count_over_time(
    {app="diabetes-prediction"} 
    | json 
    | prediction_result != "" [1h]
  )
)
```

## 🎨 Dashboard Panels to Create

### 1. Overview Panel
- **Type**: Stat
- **Query**: Total log entries
- **Time**: Last 24h

### 2. Error Rate Panel
- **Type**: Time series
- **Query**: Error rate per minute
- **Alert**: > 5%

### 3. Latency Panel
- **Type**: Gauge
- **Query**: P95 latency
- **Thresholds**: 
  - Green: < 50ms
  - Yellow: 50-100ms
  - Red: > 100ms

### 4. Prediction Distribution
- **Type**: Pie chart
- **Query**: Count by prediction_result
- **Time**: Last 1h

### 5. Log Stream
- **Type**: Logs
- **Query**: Recent logs
- **Limit**: 100

## ⚠️ Troubleshooting

### Logs Not Appearing

**Check 1: Verify credentials**
```bash
# Test with curl
curl -u "INSTANCE_ID:API_KEY" \
  -X POST \
  "YOUR_LOKI_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "streams": [
      {
        "stream": {"app": "test"},
        "values": [["'$(date +%s)'000000000", "test message"]]
      }
    ]
  }'
```

**Check 2: Environment variables loaded**
```python
import os
print(os.getenv('LOKI_URL'))
print(os.getenv('LOKI_USERNAME'))
print(os.getenv('LOKI_API_KEY'))  # Should show value
```

**Check 3: Check app logs**
```bash
# Should see: "Loki logging enabled (Grafana Cloud)"
streamlit run app.py
```

### Connection Errors

**Error: "401 Unauthorized"**
- Check API key is correct
- Verify instance ID (username)
- Regenerate API key if needed

**Error: "Connection timeout"**
- Check internet connection
- Verify Loki URL is correct
- Check firewall settings

**Error: "SSL certificate verification failed"**
```python
# In logger.py, add:
self.session.verify = True  # Ensure SSL verification
```

### No Data in Grafana

1. **Check time range** in Grafana (top right)
2. **Verify data source** is configured correctly
3. **Test query** in Explore first
4. **Check label matchers** in query

## 📈 Best Practices

### 1. Log Retention
- Free tier: 15 days retention
- Set appropriate METRICS_RETENTION_HOURS
- Export important metrics before expiry

### 2. Query Optimization
- Use label filters: `{app="diabetes-prediction", level="error"}`
- Limit time ranges: `[5m]`, `[1h]`
- Use aggregations: `rate()`, `count_over_time()`

### 3. Cost Management
- Free tier limits:
  - 50 GB logs/month
  - 10,000 series
  - 50 GB traces/month
- Monitor usage in Cloud Portal
- Set up usage alerts

### 4. Security
- ✅ Never commit API keys to git
- ✅ Use environment variables
- ✅ Rotate API keys periodically
- ✅ Use minimal permissions (MetricsPublisher)
- ✅ Enable 2FA on Grafana account

## 🔔 Setting Up Alerts

### Create Alert in Grafana Cloud

1. **Navigate to Alerting**
   - Go to **Alerting** → **Alert rules**
   - Click **New alert rule**

2. **Configure Alert: High Error Rate**
   ```
   Name: High Error Rate
   Query: 
     sum(rate({app="diabetes-prediction", level="error"}[5m]))
   Condition: WHEN last() OF query(A) IS ABOVE 0.05
   ```

3. **Set Notification**
   - Email, Slack, PagerDuty, etc.
   - Test notification
   - Save alert

### Example Alerts

**High Latency:**
```logql
quantile_over_time(0.95, 
  {app="diabetes-prediction"} 
  | json 
  | unwrap duration_ms [5m]
) > 100
```

**No Recent Activity:**
```logql
count_over_time({app="diabetes-prediction"}[10m]) < 1
```

**High Failure Rate:**
```logql
sum(rate({app="diabetes-prediction"} |= "Prediction failed"[5m])) > 0.1
```

## 📚 Additional Resources

- **Grafana Cloud Docs**: https://grafana.com/docs/grafana-cloud/
- **Loki Query Language**: https://grafana.com/docs/loki/latest/logql/
- **LogQL Tutorial**: https://grafana.com/docs/loki/latest/getting-started/logql/
- **Grafana Dashboards**: https://grafana.com/grafana/dashboards/
- **Community Forums**: https://community.grafana.com/

## ✅ Checklist

- [ ] Created Grafana Cloud account
- [ ] Obtained Loki URL
- [ ] Generated API key
- [ ] Found instance ID
- [ ] Created `.env` file
- [ ] Configured credentials
- [ ] Tested connection
- [ ] Verified logs in Grafana
- [ ] Created first dashboard
- [ ] Set up at least one alert
- [ ] Secured API credentials

## 🎉 Success!

You now have:
- ✅ Logs streaming to Grafana Cloud
- ✅ Real-time monitoring dashboard
- ✅ No infrastructure to manage
- ✅ Professional-grade observability

**Next Steps:**
1. Create custom dashboards
2. Set up alerts for critical metrics
3. Share dashboards with team
4. Explore advanced LogQL queries
5. Monitor model performance trends

---

**Need Help?**
- Check logs: `logs/diabetes-prediction.log`
- Test connection in monitoring app
- Review Grafana Cloud documentation
- Contact support: support@grafana.com
