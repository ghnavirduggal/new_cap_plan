import type { IconName } from "../_components/Icon";

export const forecastSteps: Array<{
  slug: string;
  label: string;
  icon: IconName;
  description: string;
}> = [
  {
    slug: "volume-summary",
    label: "Volume Summary",
    icon: "dashboard",
    description: "Upload volume and IQ inputs, review seasonality, and run Phase 1/2 forecasts."
  },
  {
    slug: "transformation-projects",
    label: "Transformation Projects",
    icon: "settings",
    description: "Apply sequential adjustments and publish final monthly forecasts."
  },
  {
    slug: "daily-interval",
    label: "Daily & Interval Forecast",
    icon: "calendar",
    description: "Split monthly totals into daily and interval targets for planning."
  }
];

export const forecastModels: Array<{
  title: string;
  icon: IconName;
  description: string;
  equations: string[];
}> = [
  {
    title: "Random Forest",
    icon: "layers",
    description: "Aggregates many decision trees for prediction.",
    equations: ["f(x) = (1 / B) * sum_k T_k(x)"]
  },
  {
    title: "Prophet",
    icon: "calendar",
    description: "Handles trend, seasonality, and holidays.",
    equations: ["y_t = g_t + s_t + h_t + e_t"]
  },
  {
    title: "XGBoost",
    icon: "bolt",
    description: "Gradient boosting framework for high performance.",
    equations: ["y_hat_i = sum_k f_k(x_i)", "Obj(theta) = sum l(y_i, y_hat_i) + sum Omega(f_k)"]
  },
  {
    title: "ARIMA",
    icon: "dashboard",
    description: "Combines autoregression and moving average.",
    equations: [
      "AR(p): y_t = c + phi_1 y_{t-1} + ... + phi_p y_{t-p}",
      "MA(q): y_t = alpha + theta_1 e_{t-1} + ... + theta_q e_{t-q}",
      "ARIMA(p,d,q): difference d times then ARMA(p,q)"
    ]
  },
  {
    title: "Triple Exponential Smoothing (Holt-Winters)",
    icon: "trend-down",
    description: "Captures level, trend, and seasonality.",
    equations: [
      "Level: l_t = alpha * y_t + (1 - alpha) * (l_{t-1} + b_{t-1})",
      "Trend: b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}",
      "Seasonality: s_t = gamma * (y_t / l_t) + (1 - gamma) * s_{t-m}",
      "Forecast: y_{t+1} = l_t + b_t + s_{t+1}"
    ]
  },
  {
    title: "Double Exponential Smoothing (Holt's)",
    icon: "dashboard",
    description: "Captures level and trend.",
    equations: [
      "Level: l_t = alpha * y_t + (1 - alpha) * l_{t-1}",
      "Trend: b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}",
      "Forecast: y_{t+1} = l_t + b_t"
    ]
  },
  {
    title: "Single Exponential Smoothing",
    icon: "trend-up",
    description: "Simple smoothing method.",
    equations: ["l_t = alpha * y_t + (1 - alpha) * l_{t-1}", "Forecast: y_{t+1} = l_t"]
  },
  {
    title: "Linear Regression",
    icon: "forecast",
    description: "Predicts using a linear combination of features.",
    equations: ["y_hat = beta_0 + beta_1 x_1 + ... + beta_k x_k", "RSS = sum (y_i - y_hat_i)^2"]
  },
  {
    title: "Weighted Moving Average",
    icon: "trend-up",
    description: "Forecasts using a weighted average of past observations.",
    equations: ["y_hat_t = sum (w_i * y_{t-i}), where sum w_i = 1"]
  }
];
