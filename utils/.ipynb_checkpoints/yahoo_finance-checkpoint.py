from llama_index.core.tools.tool_spec.base import BaseToolSpec
import yfinance as yf
import pandas as pd

class YahooFinanceToolSpec(BaseToolSpec):
    """Yahoo Finance tool spec."""

    spec_functions = [
        "balance_sheet",
        "income_statement",
        "cash_flow",
        "stock_basic_info",
        "stock_analyst_recommendations",
        "stock_news",
    ]

    def __init__(self) -> None:
        """Initialize the Yahoo Finance tool spec."""

    def balance_sheet(self, ticker: str, lookup: str) -> str:
        """
        Return the balance sheet of the stock.

        Args:
          ticker (str): the stock ticker to be given to yfinance
          lookup (str): category to look up. select on ['Ordinary Shares Number', 'Share Issued', 'Net Debt', 'Total Debt', 'Tangible Book Value', 'Invested Capital', 'Working Capital',
       'Net Tangible Assets', 'Capital Lease Obligations','Common Stock Equity', 'Total Capitalization','Total Equity Gross Minority Interest', 'Stockholders Equity','Gains Losses Not Affecting Retained Earnings',
       'Other Equity Adjustments', 'Retained Earnings',
       'Additional Paid In Capital', 'Capital Stock', 'Common Stock',
       'Total Liabilities Net Minority Interest',
       'Total Non Current Liabilities Net Minority Interest',
       'Other Non Current Liabilities', 'Non Current Deferred Liabilities',
       'Non Current Deferred Revenue',
       'Long Term Debt And Capital Lease Obligation', 'Long Term Debt',
       'Long Term Provisions', 'Current Liabilities',
       'Current Deferred Liabilities', 'Current Deferred Revenue',
       'Current Debt And Capital Lease Obligation',
       'Current Capital Lease Obligation', 'Current Debt',
       'Other Current Borrowings',
       'Pensionand Other Post Retirement Benefit Plans Current',
       'Current Provisions', 'Payables And Accrued Expenses',
       'Current Accrued Expenses', 'Payables', 'Total Tax Payable',
       'Income Tax Payable', 'Accounts Payable', 'Total Assets',
       'Total Non Current Assets', 'Other Non Current Assets',
       'Non Current Deferred Assets', 'Non Current Deferred Taxes Assets',
       'Goodwill And Other Intangible Assets', 'Other Intangible Assets',
       'Goodwill', 'Net PPE', 'Accumulated Depreciation', 'Gross PPE',
       'Leases', 'Construction In Progress', 'Other Properties',
       'Machinery Furniture Equipment', 'Buildings And Improvements',
       'Land And Improvements', 'Properties', 'Current Assets',
       'Other Current Assets', 'Restricted Cash', 'Prepaid Assets',
       'Inventory', 'Finished Goods', 'Raw Materials', 'Receivables',
       'Accounts Receivable', 'Allowance For Doubtful Accounts Receivable',
       'Gross Accounts Receivable',
       'Cash Cash Equivalents And Short Term Investments',
       'Other Short Term Investments', 'Cash And Cash Equivalents']

        """
        stock = yf.Ticker(ticker)
        balance_sheet = pd.DataFrame(stock.balance_sheet)
        balance_sheet = balance_sheet[balance_sheet.index == lookup]
        return "Balance Sheet: \n" + balance_sheet.to_string()

    def income_statement(self, ticker: str) -> str:
        """
        Return the income statement of the stock.

        Args:
          ticker (str): the stock ticker to be given to yfinance

        """
        stock = yf.Ticker(ticker)
        income_statement = pd.DataFrame(stock.income_stmt)
        return "Income Statement: \n" + income_statement.to_string()

    def cash_flow(self, ticker: str) -> str:
        """
        Return the cash flow of the stock.

        Args:
          ticker (str): the stock ticker to be given to yfinance

        """
        stock = yf.Ticker(ticker)
        cash_flow = pd.DataFrame(stock.cashflow)
        return "Cash Flow: \n" + cash_flow.to_string()

    def stock_basic_info(self, ticker: str) -> str:
        """
        Return the basic info of the stock. Ex: price, description, name.

        Args:
          ticker (str): the stock ticker to be given to yfinance

        """
        stock = yf.Ticker(ticker)
        return "Info: \n" + str(stock.info)

    def stock_analyst_recommendations(self, ticker: str) -> str:
        """
        Get the analyst recommendations for a stock.

        Args:
          ticker (str): the stock ticker to be given to yfinance

        """
        stock = yf.Ticker(ticker)
        return "Recommendations: \n" + str(stock.recommendations)

    def stock_news(self, ticker: str) -> str:
        """
        Get the most recent news titles of a stock.

        Args:
          ticker (str): the stock ticker to be given to yfinance

        """
        stock = yf.Ticker(ticker)
        news = stock.news
        out = "News: \n"
        for i in news:
            out += i["title"] + "\n"
        return out