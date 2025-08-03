from antback import new_cross_func
import numpy as np
import matplotlib.pyplot as plt

def simple_moving_average(data, window):
    """Calculate simple moving average"""
    if len(data) < window:
        return np.array([])

    result = []
    for i in range(window - 1, len(data)):
        avg = np.mean(data[i - window + 1 : i + 1])
        result.append(avg)

    return np.array(result)

def generate_stock_data(n_points=500, seed=42):
    """Generate realistic stock-like price data"""
    np.random.seed(seed)

    # Start with base price
    base_price = 100
    prices = [base_price]

    # Generate price movements with trend and volatility
    for i in range(1, n_points):
        # Random walk with slight upward bias
        change = np.random.normal(0.05, 1.2)  # Small upward drift, moderate volatility

        # Add some mean reversion
        if prices[-1] > base_price * 1.1:
            change -= 0.3  # Pull down if too high
        elif prices[-1] < base_price * 0.9:
            change += 0.3  # Pull up if too low

        new_price = max(prices[-1] + change, 1)  # Ensure price stays positive
        prices.append(new_price)

    return np.array(prices)


def test_cross_detection():
    """Test the cross detection function with moving averages on stock data"""
    print("=" * 60)
    print("CROSS DETECTION TEST WITH MOVING AVERAGES")
    print("=" * 60)

    # Generate stock-like data
    prices = generate_stock_data(250, seed=123)

    # Calculate moving averages
    fast_ma = simple_moving_average(prices, 5)  # 5-period MA
    slow_ma = simple_moving_average(prices, 15)  # 15-period MA

    # Align arrays (both MAs start from index 14 due to slow MA requirement)
    start_idx = 14  # slow_ma needs 15 points, so starts at index 14
    aligned_prices = prices[start_idx:]
    aligned_fast = fast_ma[10:]  # fast_ma starts at index 4, so offset by 10 more
    aligned_slow = slow_ma

    # Create cross detector
    cross_func = new_cross_func()
    cross_func2 = new_cross_func()

    # Test the cross detection
    print(f"{'Day':<4} {'Price':<8} {'Fast MA':<8} {'Slow MA':<8} {'Cross':<6}")
    print("-" * 50)

    crosses = []
    for i in range(len(aligned_fast)):
        cross_result = cross_func(aligned_fast[i], aligned_slow[i])
        cross_result2 = cross_func2(aligned_fast[i], aligned_slow[i])
        crosses.append(cross_result)

        day = start_idx + i + 1
        print(
          cross_result2,  f"{day:<4} {aligned_prices[i]:<8.2f} {aligned_fast[i]:<8.2f} {aligned_slow[i]:<8.2f} {str(cross_result):<6}"
        )

        # Highlight crosses
        if cross_result in ["up", "down"]:
            print(f"     >>> {cross_result.upper()} CROSS DETECTED! <<<")

    # Summary
    up_crosses = sum(1 for x in crosses if x == "up")
    down_crosses = sum(1 for x in crosses if x == "down")

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Total Up crosses (Fast MA above Slow MA): {up_crosses}")
    print(f"Total Down crosses (Fast MA below Slow MA): {down_crosses}")
    print(f"Total data points analyzed: {len(crosses)}")

    # Create visualization
    plt.figure(figsize=(12, 8))

    # Plot price and moving averages
    days = range(start_idx + 1, start_idx + len(aligned_fast) + 1)
    plt.subplot(2, 1, 1)
    plt.plot(days, aligned_prices, "k-", label="Price", alpha=0.7, linewidth=1)
    plt.plot(days, aligned_fast, "b-", label="Fast MA (5)", linewidth=2)
    plt.plot(days, aligned_slow, "r-", label="Slow MA (15)", linewidth=2)

    # Mark crosses
    for i, cross in enumerate(crosses):
        if cross == "up":
            plt.scatter(
                days[i], aligned_fast[i], color="green", s=100, marker="^", zorder=5
            )
        elif cross == "down":
            plt.scatter(
                days[i], aligned_fast[i], color="red", s=100, marker="v", zorder=5
            )

    plt.title("Stock Price with Moving Averages and Cross Signals")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot cross signals
    plt.subplot(2, 1, 2)
    cross_values = [1 if x == "up" else (-1 if x == "down" else 0) for x in crosses]
    plt.bar(
        days,
        cross_values,
        color=[
            "green" if x > 0 else ("red" if x < 0 else "gray") for x in cross_values
        ],
    )
    plt.title("Cross Signals (Green=Up Cross, Red=Down Cross)")
    plt.ylabel("Signal")
    plt.xlabel("Day")
    plt.ylim(-1.5, 1.5)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return prices, aligned_fast, aligned_slow, crosses


def simple_test():
    """Simple test with known values to verify logic"""
    print("=" * 40)
    print("SIMPLE TEST")
    print("=" * 40)

    # Test data where we know crosses should occur
    passive = [5, 5, 4, 3, 3, 2, 2, 3, 4, 5]
    active = [3, 4, 5, 5, 4, 4, 3, 2, 3, 4]

    cross_detector = new_cross_func()

    print(f"{'Step':<4} {'Active':<6} {'Passive':<8} {'Result':<8}")
    print("-" * 30)

    for i, (a, p) in enumerate(zip(active, passive)):
        result = cross_detector(a, p)
        print(f"{i + 1:<4} {a:<6} {p:<8} {str(result):<8}")


if __name__ == "__main__":
    # Run simple test first
    simple_test()
    print("\n")

    # Run comprehensive test with stock data
    test_cross_detection()