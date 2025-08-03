from antback import new_multi_ticker_wait, new_wait_n_bars


def test_nbars():
    """Test the new_wait_n_bars function"""
    import datetime
    print("=== Testing new_wait_n_bars function ===")

    # Test invalid n values
    print("\nTesting invalid n values:")
    try:
        new_wait_n_bars(0)
    except AssertionError as e:
        print(f"Caught invalid n=0: {e}")

    try:
        new_wait_n_bars(-1)
    except AssertionError as e:
        print(f"Caught invalid n=-1: {e}")

    # Test n=1 case
    print("\nTesting n=1 case:")
    wait1bar = new_wait_n_bars(1)
    wait1bar(start=True)  # Should return 0
    for i in range(3):
        date = datetime.date.today() + datetime.timedelta(days=i)
        result = wait1bar(bar=date)
        print(f'Bar {i} ({date}): Result={result}')
        if i == 0:
            assert result is True
        else:
            assert result is None

    # Test n=3 case
    print("\nTesting n=3 case:")
    wait3bars = new_wait_n_bars(3)
    base_date = datetime.date.today()
    test_dates = [base_date + datetime.timedelta(days=i) for i in range(18)]

    for i, date in enumerate(test_dates):
        if i in (2, 7, 12):
            print(f'Bar {i} ({date}): Starting counting (result=0)')
            wait3bars(start=True)
        elif i == 4:  # Test restart while counting
            print(f'Bar {i} ({date}): Attempting restart while counting...')
            try:
                wait3bars(start=True)
            except Exception as e:
                print(e)
        else:
            ready = wait3bars(bar=date)
            if ready is True:
                print(f'Bar {i} ({date}): *** TRIGGER ACTIVATED *** (result={ready})')
            elif ready is False:
                print(f'Bar {i} ({date}): WAITING (result={ready})')
            else:
                print(f'Bar {i} ({date}): INACTIVE (result={ready})')

    print()


def test_multiwait():
    """Test the multiwait function"""
    import datetime
    print("=== Testing multiwait function ===")

    # Test invalid n values
    print("\nTesting invalid n values:")
    try:
        new_multi_ticker_wait(0)
    except AssertionError as e:
        print(f"Caught invalid n=0: {e}")

    try:
        new_multi_ticker_wait(-1)
    except AssertionError as e:
        print(f"Caught invalid n=-1: {e}")

    # Test with n=3
    print("\nTesting multiwait with n=3:")
    mwait = new_multi_ticker_wait(3)
    base_date = datetime.date.today()
    test_dates = [base_date + datetime.timedelta(days=i) for i in range(15)]

    for i, date in enumerate(test_dates):
        # Process each ticker
        wait_a = mwait('AAPL', bar=date)
        wait_b = mwait('GOOGL', bar=date)
        wait_c = mwait('MSFT', bar=date)

        # Start triggers at different times
        if i == 1:
            print(f'Bar {i} ({date}): Starting AAPL counting')
            mwait('AAPL', start=True)
        elif i == 3:  # Test restart AAPL while counting
            print(f'Bar {i} ({date}): Restarting AAPL counting while active...')
            try:
                mwait('AAPL', start=True)
            except Exception as e:
                print('! ', e)
        elif i == 6:
            print(f'Bar {i} ({date}): Starting GOOGL counting')
            mwait('GOOGL', start=True)
        elif i == 9:
            print(f'Bar {i} ({date}): Starting MSFT counting')
            mwait('MSFT', start=True)

        # Format results
        results = [wait_a, wait_b, wait_c]
        triggered = any(result[1] is True for result in results)

        if triggered:
            # Show which tickers triggered
            trigger_info = [f"{ticker}" for ticker, result in results if result is True]
            print(f'Bar {i} ({date}): *** TRIGGERS: {", ".join(trigger_info)} *** {results}')
        else:
            print(f'Bar {i} ({date}): {results}')
    print()


def run_all_tests():
    """Run all test functions"""
    test_nbars()
    test_multiwait()


if __name__ == "__main__":
    run_all_tests()
