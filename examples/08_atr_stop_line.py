import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter
import yfinance as yf
import antback as ab


def plot_candlestick_with_stopline(df):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Convert index to matplotlib dates
    dates = mdates.date2num(df.index.to_pydatetime())

    # Candlestick width
    width = 0.5  # Width in days

    # Define colors for up and down candles
    up = df['Close'] >= df['Open']
    down = df['Close'] < df['Open']

    # Plot candlesticks
    # Up candles (green)
    ax.bar(dates[up],
           df.loc[up, 'Close'] - df.loc[up, 'Open'],
           width,
           bottom=df.loc[up, 'Open'],
           color='green')
    ax.bar(dates[up],
           df.loc[up, 'High'] - df.loc[up, 'Close'],
           width / 3,
           bottom=df.loc[up, 'Close'],
           color='green')
    ax.bar(dates[up],
           df.loc[up, 'Open'] - df.loc[up, 'Low'],
           width / 3,
           bottom=df.loc[up, 'Low'],
           color='green')

    # Down candles (red)
    ax.bar(dates[down],
           df.loc[down, 'Open'] - df.loc[down, 'Close'],
           width,
           bottom=df.loc[down, 'Close'],
           color='red')
    ax.bar(dates[down],
           df.loc[down, 'High'] - df.loc[down, 'Open'],
           width / 3,
           bottom=df.loc[down, 'Open'],
           color='red')
    ax.bar(dates[down],
           df.loc[down, 'Close'] - df.loc[down, 'Low'],
           width / 3,
           bottom=df.loc[down, 'Low'],
           color='red')

    # --- STOP LINE PLOTTING (ONLY WHEN BELOW CLOSE) ---
    if 'stop_line' in df.columns:
        # Create a copy of stop_line series where we mask values above close
        conditional_stop = df['stop_line'].where(df['stop_line'] < df['Close'])

        # Find continuous segments where stop is below close
        below_segments = []
        current_segment = []

        for i in range(len(df)):
            if pd.notna(conditional_stop.iloc[i]):
                current_segment.append(i)
            elif current_segment:
                below_segments.append(current_segment)
                current_segment = []

        # Add the last segment if it exists
        if current_segment:
            below_segments.append(current_segment)

        # Plot each continuous segment separately
        for segment in below_segments:
            segment_dates = dates[segment]
            segment_stops = conditional_stop.iloc[segment]
            ax.plot(segment_dates, segment_stops, color='blue', linestyle='--', linewidth=1)

        # Add label only once
        if below_segments:
            ax.plot([], [],
                    color='blue',
                    linestyle='--',
                    linewidth=1,
                    label='Stop Line (Below Close Only)')

    # Formatting
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    # Set y-axis limits
    # Ensure y-axis includes both price and stop line values
    y_min_price = df['Low'].min()
    y_max_price = df['High'].max()
    
    if 'stop_line' in df.columns and not df['stop_line'].isnull().all():
        y_min_stop = df['stop_line'].min(skipna=True)
        y_max_stop = df['stop_line'].max(skipna=True)
        y_min = min(y_min_price, y_min_stop)
        y_max = max(y_max_price, y_max_stop)
    else:
        y_min = y_min_price
        y_max = y_max_price
        
    ax.set_ylim(y_min * 0.99, y_max * 1.01)


    # Labels and title
    ax.set_title('Candlestick Chart with Stop Line', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to fetch stock data, calculate ATR stop loss, and plot the results.
    """
    # -- Configuration --
    # Create an ATR (Average True Range) stop function with a period of 8 and a multiplier of 4.
    # This function will maintain its state between calls.
    atr_stop_calculator = ab.new_atr_stop_function(n=8, atr_multiplier=4)
    
    # Define the stock symbol and data period.
    ticker_symbol = "QQQ"
    data_period = '10y'
    # Create a Ticker object for the specified symbol.
    ticker_object = yf.Ticker(ticker_symbol)
    stock_data_df = ticker_object.history(period=data_period)

    # Define a helper function to apply the ATR calculator to each row of the DataFrame.
    def calculate_atr_stop_for_row(row):
        # The calculator takes High, Low, and Close prices and returns the ATR state and the stop line value.
        atr_state, stop_line_value = atr_stop_calculator(row['High'], row['Low'], row['Close'])
        return atr_state, stop_line_value

    # Apply the helper function across all rows to generate the 'atr_state' and 'stop_line' columns.
    # 'axis=1' applies the function row-by-row.
    # 'result_type='expand'' splits the tuple returned by the function into separate new columns.
    stock_data_df[['atr_state', 'stop_line']] = stock_data_df.apply(
        calculate_atr_stop_for_row, axis=1, result_type='expand'
    )
    
    # Select the last 200 trading days for a more focused visualization.
    data_for_plotting = stock_data_df[['Open', 'High', 'Low', 'Close', 'stop_line']].tail(200)

    # Print the calculated values for inspection or debugging.
    print("Calculated ATR State and Stop Line:")
    print(stock_data_df[['Close', 'atr_state', 'stop_line']])
    print("\n--- Plotting last 200 days ---")

    # Call the plotting function with the prepared data.
    plot_candlestick_with_stopline(data_for_plotting)


if __name__ == "__main__":
    main()