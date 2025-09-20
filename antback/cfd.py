#!/usr/bin/python
# coding=utf-8
from collections import defaultdict, namedtuple
from datetime import date, datetime
from math import isnan
import numpy as np
import pandas as pd

try:
    from . import utils
except ImportError:
    import utils


def pct_diff(prev, today):
    if prev == 0:
        return float('nan')
    return (today - prev) / prev * 100


TradeData = namedtuple(
    "TradeData",
    [
        "ticker",
        "open_date",
        "open_price",
        "close_date",
        "close_price",
        "quantity",
        "position_type",  # 'long' or 'short'
        "log_msg",
    ],
)

TradeResult = namedtuple(
    "TradeResult",
    [
        "ticker",
        "open_date",
        "close_date",
        "open_price",
        "close_price",
        "quantity",
        "position_type",
        "position_size",
        "net_profit",
        "profit_pct",
        "gross_profit",
        "buy_fees",
        "sell_fees",
        "total_fees",
        "log_msg",
    ],
)


def trade_result(trade, brokerage, opened=False):
    """Calculate trade results for CFD/FX positions."""
    position_size = trade.open_price * trade.quantity

    if trade.position_type == "long":
        # Long CFD: profit when price goes up
        sell_value = trade.close_price * trade.quantity
        gross_profit = sell_value - position_size
        profit_pct = pct_diff(trade.open_price, trade.close_price)
    else:  # short
        # Short CFD: profit when price goes down
        cover_cost = trade.close_price * trade.quantity
        gross_profit = position_size - cover_cost
        profit_pct = pct_diff(trade.close_price, trade.open_price)

    # In CFD/FX, fees are typically spread-based or commission per trade
    buy_fees = position_size * brokerage
    sell_fees = 0 if opened else (trade.close_price * trade.quantity * brokerage)
    total_fees = buy_fees + sell_fees
    net_profit = gross_profit - total_fees

    return TradeResult(
        ticker=trade.ticker,
        open_date=trade.open_date,
        close_date=trade.close_date,
        open_price=trade.open_price,
        close_price=trade.close_price,
        quantity=trade.quantity,
        position_type=trade.position_type,
        position_size=position_size,
        net_profit=net_profit,
        profit_pct=profit_pct,
        gross_profit=gross_profit,
        buy_fees=buy_fees,
        sell_fees=sell_fees,
        total_fees=total_fees,
        log_msg="*opened" if opened else trade.log_msg,
    )


class CFDAccount:
    """CFD/FX Trading portfolio with margin requirements and leverage."""

    __slots__ = (
        'warn', 'starting_capital', 'cash', 'blocked_margin', 'position',
        'margin_requirement', 'leverage', 'margin_call_level', 'fees',
        'max_date', 'trades', 'pos_history', 'events', 'total_fees_paid',
        'allow_fractional'
    )

    DEFAULT_PERCENT = 1.0
    MIN_AMOUNT = 500

    def __init__(
        self,
        cash,
        margin_requirement=0.1,
        leverage=2,
        warn=False,
        allow_fractional=True,
        fees=0.00015,
        margin_call_level=0.5,
    ):
        """
        Initialize CFD/FX portfolio.

        Args:
            cash: Starting capital
            margin_requirement: Required margin as fraction (0.1 = 10%)
            leverage: Maximum leverage ratio
            fees: Fee rate (default 0.00015 = 1.5 bps)
            margin_call_level: Margin call triggered when equity falls below this * required_margin
        """
        self.warn = warn
        assert cash >= 1000, "Minimum cash requirement: 1,000 for CFD/FX"

        self.starting_capital = cash
        self.cash = cash  # Available cash
        self.blocked_margin = 0.0  # Cash blocked as margin for open positions
        self.position = None  # Will become a list when position exists

        # CFD/FX specific parameters
        self.margin_requirement = margin_requirement
        self.leverage = leverage
        self.margin_call_level = margin_call_level

        self.fees = fees
        self.max_date = None
        self.trades = []
        self.pos_history = {}
        self.events = defaultdict(list)
        self.total_fees_paid = 0.0
        self.allow_fractional = allow_fractional

    def _normalize_date(self, date_obj):
        if isinstance(date_obj, (date, datetime)):
            return date_obj
        return pd.to_datetime(date_obj)

    def _validate_and_update_date(self, date_obj):
        assert isinstance(date_obj, (date, datetime))

        if self.max_date and date_obj < self.max_date:
            print(f"! Received date: {date_obj}, self.max_date: {self.max_date}")
            raise ValueError("Date not in chronological order")

        if not self.max_date or date_obj > self.max_date:
            self.max_date = date_obj

        return date_obj

    def available_cash(self):
        """Calculate available cash (total cash minus blocked margin)."""
        return self.cash - self.blocked_margin

    def calculate_unrealized_pnl(self):
        """Calculate unrealized P&L for open position."""
        if not self.position:
            return 0.0

        ticker, quantity, entry_price, entry_date, current_price, position_type = self.position

        if position_type == "long":
            # Long position: profit when price goes up
            unrealized_pnl = (current_price - entry_price) * quantity
        else:  # short position
            # Short position: profit when price goes down
            unrealized_pnl = (entry_price - current_price) * quantity

        return unrealized_pnl

    def check_margin_call(self):
        if not self.position:
            return False

        unrealized_pnl = self.calculate_unrealized_pnl()

        # FIXED: Use total equity (cash + unrealized_pnl) instead of available_cash
        total_equity = self.cash + unrealized_pnl
        required_margin = self.blocked_margin

        # Margin call when equity falls below margin call level
        margin_call_threshold = required_margin * self.margin_call_level

        if total_equity < margin_call_threshold:
            return True, total_equity, margin_call_threshold
        return False, total_equity, margin_call_threshold

    def _open_position(self, ticker, date_obj, price, fixed_val, position_type):
        """Open a CFD/FX position with margin requirements."""
        if self.position is not None:
            opened_position_type = self.position[-1]
            if opened_position_type == position_type:
                if self.warn:
                    print(f"- Already have position in {self.position[0]}.")
                return
            else:
                raise Exception(
                    f"have opposite position [{self.position[-1]}] in {self.position}. Close it first."
                )

        date_obj = self._validate_and_update_date(date_obj)
        # Calculate position size
        if fixed_val:
            assert fixed_val >= self.MIN_AMOUNT, (f"fixed_val amount must be >= {
                self.MIN_AMOUNT}")
            position_value = fixed_val
        else:
            # Use available cash with leverage
            max_position_value = self.available_cash() * self.leverage
            position_value = max_position_value * self.DEFAULT_PERCENT

        # Calculate required margin
        required_margin = position_value * self.margin_requirement

        if self.available_cash() < required_margin:
            msg = f"Insufficient margin: available {
                self.available_cash():.1f} < required {
                required_margin:.1f}"
            self.events[date_obj].append(msg)
            raise ValueError(msg)

        # Calculate quantity
        buffer_factor = 0.99999
        quantity = (position_value * buffer_factor) / price
        if not self.allow_fractional:
            quantity = int(quantity)

        if quantity <= 0.001:
            msg = f"! {position_type} quantity <= 0.001 for {ticker} at {price}"
            self.events[date_obj].append(msg)
            if self.warn:
                print(msg)
            return None

        # Recalculate actual position value based on final quantity
        actual_position_value = price * quantity
        actual_required_margin = actual_position_value * self.margin_requirement

        # Calculate fees on the position value (not just margin)
        fees = actual_position_value * self.fees

        # Block margin and pay fees
        self.blocked_margin += actual_required_margin
        self.cash -= fees  # Fees are paid immediately

        self.total_fees_paid += fees
        self.position = [ticker, quantity, price, date_obj, price, position_type]

        action_msg = (f"{position_type}: {ticker}({price:.2f}*{quantity:.1f}, "
                      f"value: {actual_position_value:.1f}, margin: {actual_required_margin:.1f}, "
                      f"fees: {fees:.1f})")
        self.events[date_obj].append(action_msg)

        if self.warn:
            print(
                f"Opened {position_type} position: {
                    quantity:.4f} {ticker} at {
                    price:.2f}")
            print(
                f"Position value: {
                    actual_position_value:.2f}, Required margin: {
                    actual_required_margin:.2f}")
            print(f"Available cash: {self.available_cash():.2f}")

        self.save_position(date_obj)
        return quantity

    def _close(self, date_obj, price, log_msg=""):
        """Close the current CFD/FX position."""
        if self.position is None:
            if self.warn:
                print("No position to close")
            return

        date_obj = self._validate_and_update_date(date_obj)
        ticker, quantity, entry_price, entry_date, _, position_type = self.position

        # Calculate P&L
        if position_type == "long":
            gross_pnl = (price - entry_price) * quantity
        else:  # short position
            gross_pnl = (entry_price - price) * quantity

        # Calculate closing fees
        position_value = price * quantity
        closing_fees = position_value * self.fees
        net_pnl = gross_pnl - closing_fees

        # Release blocked margin and apply net P&L
        released_margin = self.blocked_margin
        self.blocked_margin = 0.0
        self.cash += net_pnl

        self.total_fees_paid += closing_fees

        self.trades.append(
            TradeData(
                ticker,
                entry_date,
                entry_price,
                date_obj,
                price,
                quantity,
                position_type,
                log_msg,
            ))

        action = "close_long" if position_type == "long" else "close_short"
        pnl_desc = f"P&L: {net_pnl:.1f}" if net_pnl >= 0 else f"P&L: {net_pnl:.1f}"

        self.events[date_obj].append(
            f"{action}: {ticker}({price:.2f}*{quantity:.1f}, {pnl_desc}, fees: {closing_fees:.1f})")

        if self.warn:
            print(
                f"Closed {position_type} position: {
                    quantity:.4f} {ticker} at {
                    price:.2f}")
            print(f"Gross P&L: {gross_pnl:.2f}, Net P&L: {net_pnl:.2f}")
            print(
                f"Released margin: {
                    released_margin:.2f}, Available cash: {
                    self.available_cash():.2f}")

        self.position = None
        self.save_position(date_obj)

    def _update(self, ticker, date_obj, price):
        """Update position's current price and check for margin calls."""
        date_obj = self._validate_and_update_date(date_obj)
        if price <= 0:
            raise ValueError("Price must be positive")

        if self.position and self.position[0] == ticker:
            self.position[4] = price  # Update current price

            # Check for margin call
            is_margin_call, equity, threshold = self.check_margin_call()
            if is_margin_call:
                msg = f"MARGIN CALL: Equity {equity:.2f} < Threshold {threshold:.2f}"
                self.events[date_obj].append(msg)
                if self.warn:
                    print(f"!  {msg}")

            self.save_position(date_obj)

    def process(self, signal, ticker, date_obj, price, fixed_val=None, log_msg=""):
        """Unified entry-point for any daily signal."""
        if price <= 0 or isnan(price):
            raise ValueError("Price must be positive")
        
        if signal == "update" or signal is None:
            self._update(ticker, date_obj, price)
        elif signal == "long":
            self._open_position(ticker, date_obj, price, fixed_val, "long")
        elif signal == "short":
            self._open_position(ticker, date_obj, price, fixed_val, "short")
        elif signal == "close":
            self._close(date_obj, price, log_msg=log_msg)
        else:
            raise ValueError(f"Unknown signal {signal!r}")

    def save_position(self, record_date):
        """Save current portfolio state with entry_price for proper unrealized P&L calculation."""
        if self.position:
            ticker, quantity, entry_price, entry_date, current_price, position_type = (
                self.position)
            pos_data = (
                ticker,
                quantity,
                entry_price,
                current_price,
                position_type,
                self.blocked_margin,
            )
        else:
            pos_data = None

        self.pos_history[record_date] = (self.cash, self.blocked_margin, pos_data)

    def has_position(self, ticker=None):
        """Check if there's a current position, optionally for specific ticker."""
        if ticker is None:
            return self.position is not None
        return self.position is not None and self.position[0] == ticker

    def get_open_position(self):
        """Get information about the current open position."""
        if not self.position:
            return {"unrealized_pnl": 0, "open_trade": None, "margin_used": 0}

        ticker, quantity, entry_price, entry_date, current_price, position_type = (
            self.position)

        trade_data = TradeData(
            ticker=ticker,
            open_date=entry_date,
            open_price=entry_price,
            close_date=self.max_date or datetime.now(),
            close_price=current_price,
            quantity=quantity,
            position_type=position_type,
            log_msg="open",
        )

        open_trade = trade_result(trade_data, self.fees, opened=True)
        unrealized_pnl = self.calculate_unrealized_pnl()

        return {
            "unrealized_pnl": unrealized_pnl,
            "open_trade": open_trade,
            "margin_used": self.blocked_margin,
        }

    def current_value(self):
        """Calculate current total portfolio value (equity)."""
        unrealized_pnl = self.calculate_unrealized_pnl()

        # CORRECT: Use total cash, not just available cash.
        # Blocked margin is part of the total cash.
        total_equity = self.cash + unrealized_pnl

        if self.warn and self.position:
            print(
                f"Total Cash: {
                    self.cash:.2f}, Blocked Margin: {
                    self.blocked_margin:.2f}")
            print(
                f"Unrealized P&L: {
                    unrealized_pnl:.2f}, Total Equity: {
                    total_equity:.2f}")

        return total_equity

    def get_tradelist(self):
        """Returns DataFrame of all trades."""
        closed_trades = [trade_result(trade, self.fees) for trade in self.trades]

        open_trade_info = self.get_open_position()
        open_trades = ([open_trade_info["open_trade"]]
                       if open_trade_info["open_trade"] else [])

        all_trades = closed_trades + open_trades
        if all_trades:
            df = pd.DataFrame(all_trades, columns=TradeResult._fields)
            df = df.sort_values("close_date").reset_index(drop=True)
            return df
        else:
            return pd.DataFrame(columns=TradeResult._fields)

    def margin_utilization(self):
        """Calculate margin utilization percentage."""
        if self.blocked_margin == 0:
            return 0.0
        return (self.blocked_margin / self.cash) * 100

    def __repr__(self):
        """String representation of CFD/FX portfolio."""
        print("---- CFD/FX PORTFOLIO ----")
        print(f"Total Equity: {self.current_value():.2f}")
        print(f"Available Cash: {self.available_cash():.2f}")
        print(f"Blocked Margin: {self.blocked_margin:.2f}")

        if self.position:
            ticker, quantity, entry_price, entry_date, current_price, position_type = (
                self.position)
            unrealized_pnl = self.calculate_unrealized_pnl()
            position_value = current_price * quantity

            print(
                f"Position: {
                    position_type.upper()} {
                    quantity:.4f} {ticker} @ {
                    current_price:.2f}")
            print(f"Position Value: {position_value:.2f}, Entry: {entry_price:.2f}")
            print(f"Unrealized P&L: {unrealized_pnl:.2f}")

            # Margin call check
            is_margin_call, equity, threshold = self.check_margin_call()
            if is_margin_call:
                print(f"!  MARGIN CALL: Equity {equity:.2f} < Threshold {threshold:.2f}")
        else:
            print("Position: None")

        print(f"Margin Utilization: {self.margin_utilization():.1f}%")
        print(f"Leverage Available: {self.leverage}x")
        print(f"Total Fees Paid: {self.total_fees_paid:.2f}")
        print(f"Return: {((self.current_value() / self.starting_capital - 1) * 100):.2f}%")

        return f"Last Date: {self.max_date}"

    def equity_line(self):
        """Generate equity line data for CFD/FX portfolio with proper unrealized P&L calculation."""
        dates, equity = [], []

        for date_obj, data in sorted(self.pos_history.items()):
            cash, blocked_margin, pos_data = data

            if pos_data:
                # pos_data format: (ticker, quantity, entry_price, current_price,
                # position_type, blocked_margin)
                (
                    ticker,
                    quantity,
                    entry_price,
                    current_price,
                    position_type,
                    stored_blocked_margin,
                ) = pos_data

                # Calculate unrealized P&L using entry_price vs current_price
                if position_type == "long":
                    unrealized_pnl = (current_price - entry_price) * quantity
                else:  # short position
                    unrealized_pnl = (entry_price - current_price) * quantity
            else:
                unrealized_pnl = 0

            # FIXED: Total equity = total cash + unrealized P&L
            # (blocked_margin is already part of total cash, just locked up)
            total_equity = cash + unrealized_pnl

            dates.append(date_obj)
            equity.append(total_equity)

        return dates, equity

    def verify_consistency(self):
        """Verify portfolio calculations are consistent."""
        if not self.pos_history:
            return True

        dates, capital = self.equity_line()
        if dates and capital:
            equity_line_total = capital[-1]
            current_value_total = self.current_value()

            difference = abs(equity_line_total - current_value_total)
            if difference > 0.01:
                print("WARNING: Calculation mismatch!")
                print(f"Equity line latest: {equity_line_total:.2f}")
                print(f"Current sum: {current_value_total:.2f}")
                print(f"Difference: {difference:.2f}")
                return False

        current_total = self.current_value()

        net_trading_profit = sum(
            trade_result(
                trade,
                self.fees).net_profit for trade in self.trades)

        open_positions_profit = 0.0
        if self.position:
            # self.position is [ticker, quantity, entry_price, entry_date, current_price, position_type]
            ticker, quantity, entry_price, entry_date, current_price, position_type = (
                self.position)

            if position_type == "long":
                # For long positions: profit = (current_price - entry_price) * quantity -
                # fees
                gross_profit = (current_price - entry_price) * quantity
                # Only entry fees for open position
                fees = (entry_price * quantity * self.fees)
                open_positions_profit = gross_profit - fees
            else:  # short position
                # For short positions: profit = (entry_price - current_price) * quantity -
                # fees
                gross_profit = (entry_price - current_price) * quantity
                # Only entry fees for open position
                fees = (entry_price * quantity * self.fees)
                open_positions_profit = gross_profit - fees

        if self.warn:
            print("Capital verification:")
            print(f"  Starting: {self.starting_capital:.2f}")
            print(f"  Current total: {current_total:.2f}")
            print(f"  Closed trades profit: {net_trading_profit:.2f}")
            print(f"  Open positions profit: {open_positions_profit:.2f}")
            print(f"  Total fees: {self.total_fees_paid:.2f}")

        return True

    # helper methods

    def get_equity(self, show=False):
        return gen_equity(self, show=show)

    def stats_trades(self):
        return utils.analyze_trades(self.get_tradelist(), show=False)

    def full_report(self, kind="html", outfile="report.html", title="CFDAccount report"):
        if kind == "excel":
            return utils.excel_report(self, title=title)
        return utils.html_report(self, outfile=outfile, title=title)

    def basic_report(self, show=True):
        return utils.summary(self, show=show)

    def base_results(self):
        dates, capital_line = self.equity_line()
        drawdown = utils.get_drawdown(np.array(capital_line))
        drawdown_val = drawdown["max_dd"]
        pf_return = pct_diff(self.starting_capital, self.current_value())
        return pf_return, drawdown_val

def gen_equity(portfolio, show=False):
    """Generate equity line data for CFD/FX portfolio."""

    def calculate_eqt_val(portfolio):
        outlist = []

        for date_obj in sorted(portfolio.pos_history.keys()):
            # CFD structure: (cash, blocked_margin, pos_data)
            cash, blocked_margin, pos_data = portfolio.pos_history[date_obj]

            # Calculate available cash (total cash - blocked margin)
            available_cash = cash - blocked_margin

            # Handle position data and calculate unrealized P&L
            if pos_data is None:
                positions = []
                unrealized_pnl = 0
            else:
                # CFD pos_data: (ticker, quantity, entry_price, current_price,
                # position_type, margin_used)
                (
                    ticker,
                    quantity,
                    entry_price,
                    current_price,
                    position_type,
                    margin_used,
                ) = pos_data

                positions = [(ticker, quantity, current_price, position_type)]

                # FIXED: Calculate proper unrealized P&L based on position type
                if position_type == "long":
                    # Long position: profit when price goes up
                    unrealized_pnl = (current_price - entry_price) * quantity
                else:  # short position
                    # Short position: profit when price goes down
                    unrealized_pnl = (entry_price - current_price) * quantity

            # FIXED: Total equity = total cash + unrealized P&L (blocked margin is
            # part of cash)
            capital = cash + unrealized_pnl

            # Format positions for display
            if positions:
                positions_str = ", ".join(
                    f"({ticker}={current_price:.4f}*{quantity:.1f}[{position_type}])"
                    for ticker, quantity, current_price, position_type in positions)
            else:
                positions_str = ""

            outlist.append((
                date_obj,
                capital,
                available_cash,
                blocked_margin,
                positions_str,
            ))

        return outlist

    # Check for missing events in position history
    events_history_diff = set(portfolio.events).difference(portfolio.pos_history)
    if events_history_diff:
        print(f"Warning: {len(events_history_diff)} events not in position history")

    equity = calculate_eqt_val(portfolio)
    if not equity:
        print("No equity data available")
        return pd.DataFrame()

    # DEBUG: Print first few equity calculations to verify
    if show:
        print("\n=== DEBUG: First 5 equity calculations ===")
        for i, (date, cap, avail, blocked, pos) in enumerate(equity[:5]):
            cash_data = portfolio.pos_history[date]
            pos_data = cash_data[2]  # pos_data
            if pos_data:
                ticker, qty, entry, current, pos_type, _ = pos_data
                unrealized = ((current - entry) * qty if pos_type == "long" else
                              (entry - current) * qty)
                print(
                    f"{date}: entry={
                        entry:.4f}, current={
                        current:.4f}, unrealized_pnl={
                        unrealized:.2f}, capital={
                        cap:.2f}")
            else:
                print(f"{date}: No position, capital={cap:.2f}")

    # Convert events dict
    portfolio.events = dict(portfolio.events)

    # Create DataFrame with CFD-specific columns
    mdf = pd.DataFrame(
        equity,
        columns=[
            "date_obj",
            "capital",
            "available_cash",
            "blocked_margin",
            "positions",
        ],
    )

    # Add events information
    try:
        # Assuming utils module has datetime_to_str function
        events_dict = {utils.datetime_to_str(k): v for k, v in portfolio.events.items()}
        mdf_py_dates = [utils.datetime_to_str(x) for x in mdf.date_obj.values]
        mdf["events"] = [str(events_dict.get(x, "")) for x in mdf_py_dates]
    except (NameError, AttributeError):
        # Fallback if utils module not available
        events_dict = {str(k): v for k, v in portfolio.events.items()}
        mdf_py_dates = [str(x) for x in mdf.date_obj.values]
        mdf["events"] = [str(events_dict.get(x, "")) for x in mdf_py_dates]

    # Convert to date for grouping
    mdf["pydate"] = mdf.date_obj.map(lambda x: pd.to_datetime(x).date())

    # Group by day and take last values
    gdf = mdf.groupby(["pydate"]).agg({
        "date_obj": "last",
        "capital": "last",
        "available_cash": "last",
        "blocked_margin": "last",
        "positions": "last",
        "events": lambda x: "".join(x),
    })

    # Fill missing days
    all_days = pd.date_range(mdf.pydate.min(), mdf.pydate.max(), freq="D")
    ndf = gdf.reindex(all_days)  # .ffill()
    ndf["date_obj"] = ndf.index
    ndf["capital"] = ndf["capital"].ffill()
    # Convert date_obj back to string format
    try:
        ndf["date_obj"] = ndf["date_obj"].map(lambda x: utils.datetime_to_str(x)
                                              if pd.notna(x) else x)
    except (NameError, AttributeError):
        ndf["date_obj"] = ndf["date_obj"].map(lambda x: str(x) if pd.notna(x) else x)

    if show:
        print("\n=== CFD/FX Equity Line ===")
        print(ndf[["capital", "available_cash", "blocked_margin", "positions"]].head(10))
        print(f"Total rows: {len(ndf)}")

        # Show capital changes
        print("\n=== Capital Changes ===")
        capital_changes = ndf["capital"].diff().dropna()
        non_zero_changes = capital_changes[capital_changes != 0]
        if len(non_zero_changes) > 0:
            print(f"Found {len(non_zero_changes)} days with capital changes")
            print("First 5 changes:", non_zero_changes.head().tolist())
        else:
            print("WARNING: No capital changes detected!")

    return ndf


def simple_test():
    """Simple showcase of portfolio functionality with long/short examples."""
    print("=" * 60)
    print("SIMPLE PORTFOLIO SHOWCASE")
    print("=" * 60)

    # Create portfolio with $50,000
    portfolio = CFDAccount(cash=100000, warn=True)
    print(f"Starting portfolio: ${portfolio.current_value():.2f}")
    print()

    # Example 1: Long position
    print("--- LONG POSITION EXAMPLE ---")

    portfolio.process("close", "AAPL", "2024-01-01", 120.0)

    portfolio.process("long", "AAPL", "2024-01-01", 150.0)
    print(f"After going long AAPL @ $150: ${portfolio.current_value():.2f}")

    # portfolio.process("short", "AAPL", "2024-01-01", 151.0)

    # Price goes up
    portfolio.process(None, "AAPL", "2024-01-02", 160.0)
    print(f"After AAPL rises to $160: ${portfolio.current_value():.2f}")

    # Close long position
    portfolio.process("close", "AAPL", "2024-01-03", 165.0, "@@")
    print(f"After closing long @ $165: ${portfolio.current_value():.2f}")
    print()

    # Example 2: Short position
    print("--- SHORT POSITION EXAMPLE ---")
    print(portfolio.cash)
    portfolio.process("short", "TSLA", "2024-01-04", 200.0)
    print(f"After shorting TSLA @ $200: ${portfolio.current_value():.2f}")

    # Price goes down (good for short)
    portfolio.process(None, "TSLA", "2024-01-05", 180.0)
    print(f"After TSLA falls to $180: ${portfolio.current_value():.2f}")

    # Price goes up (bad for short)
    portfolio.process(None, "TSLA", "2024-01-06", 190.0)
    print(f"After TSLA rises to $190: ${portfolio.current_value():.2f}")
    print(portfolio.cash)
    # x
    # Close short position
    portfolio.process("close", "TSLA", "2024-01-07", 185.0)
    print(f"After closing short @ $185: ${portfolio.current_value():.2f}")
    print()

    # Show final results
    print("--- FINAL RESULTS ---")
    print(portfolio)
    print()

    # Show trade history
    trades_df = portfolio.get_tradelist()
    print("--- TRADE HISTORY ---")
    print(trades_df[[
        "ticker",
        "position_type",
        "open_price",
        "close_price",
        "net_profit",
        "profit_pct",
    ]])
    print()
    portfolio.basic_report()

    portfolio.full_report()
    portfolio.full_report("excel")
    import df2tables as df2t

    df = portfolio.get_equity()
    df2t.render(df, to_file="eqt_short1.html")
    return portfolio




def comprehensive_test():
    """Comprehensive example with random stock data and multiple trades."""
    print("=" * 60)
    print("COMPREHENSIVE TRADING SIMULATION")
    print("=" * 60)

    # Create portfolio
    # Turn off warnings for cleaner output
    portfolio = CFDAccount(cash=100000, warn=0, leverage=1, fees=0.0)

    # Generate random stock data
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range("2024-01-01", periods=100, freq="D")

    # Simulate stock price with random walk
    initial_price = 100.0
    # Small daily returns with volatility
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = [initial_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    stock_data = pd.DataFrame({"date": dates, "price": prices})

    print(f"Starting portfolio value: ${portfolio.current_value():,.2f}")
    print(f"Simulating {len(dates)} days of trading...")
    print()

    # Trading strategy: Simple momentum/contrarian strategy
    equity_curve = []

    for i, row in stock_data.iterrows():
        date = row["date"]
        price = row["price"]

        # Simple trading rules
        if i == 0:
            # Start with a long position
            portfolio.process("long", "STOCK", date, price)
            signal = "LONG"
        elif i < 10:
            # Just update prices for first 10 days
            portfolio.process(None, "STOCK", date, price)
            signal = "UPDATE"
        elif i == 10:
            # Close first position
            print(date, price)
            portfolio.process("close", "STOCK", date, price)
            signal = "CLOSE"
        elif i == 15:
            # Go short
            portfolio.process("short", "STOCK", date, price)
            signal = "SHORT"
        elif i == 25:
            # Close short
            portfolio.process("close", "STOCK", date, price)
            signal = "CLOSE"
        elif i == 30:
            # Go long again
            portfolio.process("long", "STOCK", date, price)
            signal = "LONG"
        elif i == 50:
            # Close and go short
            portfolio.process("close", "STOCK", date, price)
            portfolio.process("short", "STOCK", date, price)
            signal = "CLOSE->SHORT"
        elif i == 70:
            # Close short and go long
            portfolio.process("close", "STOCK", date, price)
            portfolio.process("long", "STOCK", date, price)
            signal = "CLOSE->LONG"
        elif i == len(stock_data) - 1:
            # Close final position
            pass
            # portfolio.process("close", "STOCK", date, price)
            # signal = "CLOSE"
        else:
            # Just update price
            portfolio.process(None, "STOCK", date, price)
            signal = "UPDATE"

        # Record portfolio value
        portfolio_value = portfolio.current_value()
        equity_curve.append({
            "date": date,
            "price": price,
            "portfolio_value": portfolio_value,
            "signal": signal,
        })

        # Print key trading days
        if signal != "UPDATE":
            print(
                f"{date.strftime('%Y-%m-%d')}: {signal:12} @ ${price:6.2f} | CFDAccount: ${portfolio_value:8,.2f}"
            )

    print()
    print("--- FINAL RESULTS ---")
    print(portfolio)
    # Show trade history
    trades_df = portfolio.get_tradelist()
    print("--- COMPLETE TRADE HISTORY ---")
    print(trades_df[[
        "ticker",
        "position_type",
        "open_date",
        "close_date",
        "open_price",
        "close_price",
        "net_profit",
        "profit_pct",
    ]].round(2))

    import df2tables as df2t

    dates, eqt = portfolio.equity_line()
    edf = pd.DataFrame(zip(dates, eqt), columns=["d", "e"])
    df2t.render(edf, to_file="eqt_short.html")
    portfolio.basic_report()
    portfolio.full_report()
    # portfolio.full_report("excel")
    return portfolio, stock_data


if __name__ == "__main__":
    # Run simple test
    # simple_portfolio = simple_test()
    # Run comprehensive test
    comp_portfolio, stock_data = comprehensive_test()
