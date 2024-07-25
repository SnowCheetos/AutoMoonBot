use crate::data::*;

pub trait Aggregates: Clone + Copy + IntoIterator<Item = f64> {
    fn to_vec(&self) -> Vec<f64> {
        self.into_iter().collect()
    }
    fn timestamp(&self) -> Instant;
    fn duration(&self) -> Duration;
}

pub struct IncomeStatementIterator {
    aggregate: IncomeStatement,
    index: usize,
}

pub struct BalanceSheetIterator {
    aggregate: BalanceSheet,
    index: usize,
}

pub struct CashFlowIterator {
    aggregate: CashFlow,
    index: usize,
}

pub struct EarningsIterator {
    aggregate: Earnings,
    index: usize,
}

pub struct PriceAggregateIterator {
    aggregate: PriceAggregate,
    index: usize,
}

pub struct OptionsAggregateIterator {
    aggregate: OptionsAggregate,
    index: usize,
}

impl Iterator for IncomeStatementIterator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let result = match self.index {
            0 => Some(self.aggregate.gross_profit),
            1 => Some(self.aggregate.total_revenue),
            2 => Some(self.aggregate.cost_of_revenue),
            3 => Some(self.aggregate.cost_of_goods_and_services_sold),
            4 => Some(self.aggregate.operating_income),
            5 => Some(self.aggregate.selling_general_and_administrative),
            6 => Some(self.aggregate.research_and_development),
            7 => Some(self.aggregate.operating_expenses),
            8 => Some(self.aggregate.investment_income_net),
            9 => Some(self.aggregate.net_interest_income),
            10 => Some(self.aggregate.interest_income),
            11 => Some(self.aggregate.interest_expense),
            12 => Some(self.aggregate.non_interest_income),
            13 => Some(self.aggregate.other_non_operating_income),
            14 => Some(self.aggregate.depreciation),
            15 => Some(self.aggregate.depreciation_and_amortization),
            16 => Some(self.aggregate.income_before_tax),
            17 => Some(self.aggregate.income_tax_expense),
            18 => Some(self.aggregate.interest_and_debt_expense),
            19 => Some(self.aggregate.net_income_from_continuing_operations),
            20 => Some(self.aggregate.comprehensive_income_net_of_tax),
            21 => Some(self.aggregate.ebit),
            22 => Some(self.aggregate.ebitda),
            23 => Some(self.aggregate.net_income),
            _ => None,
        };
        self.index += 1;
        result
    }
}

impl Iterator for BalanceSheetIterator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let result = match self.index {
            0 => Some(self.aggregate.total_assets),
            1 => Some(self.aggregate.total_current_assets),
            2 => Some(self.aggregate.cash_and_cash_equivalents_at_carrying_value),
            3 => Some(self.aggregate.cash_and_short_term_investments),
            4 => Some(self.aggregate.inventory),
            5 => Some(self.aggregate.current_net_receivables),
            6 => Some(self.aggregate.total_non_current_assets),
            7 => Some(self.aggregate.property_plant_equipment),
            8 => Some(self.aggregate.accumulated_depreciation_amortization_ppe),
            9 => Some(self.aggregate.intangible_assets),
            10 => Some(self.aggregate.intangible_assets_excluding_goodwill),
            11 => Some(self.aggregate.goodwill),
            12 => Some(self.aggregate.investments),
            13 => Some(self.aggregate.long_term_investments),
            14 => Some(self.aggregate.short_term_investments),
            15 => Some(self.aggregate.other_current_assets),
            16 => Some(self.aggregate.other_non_current_assets),
            17 => Some(self.aggregate.total_liabilities),
            18 => Some(self.aggregate.total_current_liabilities),
            19 => Some(self.aggregate.current_accounts_payable),
            20 => Some(self.aggregate.deferred_revenue),
            21 => Some(self.aggregate.current_debt),
            22 => Some(self.aggregate.short_term_debt),
            25 => Some(self.aggregate.total_non_current_liabilities),
            26 => Some(self.aggregate.capital_lease_obligations),
            27 => Some(self.aggregate.long_term_debt),
            28 => Some(self.aggregate.current_long_term_debt),
            29 => Some(self.aggregate.long_term_debt_noncurrent),
            30 => Some(self.aggregate.short_long_term_debt_total),
            31 => Some(self.aggregate.other_current_liabilities),
            32 => Some(self.aggregate.other_non_current_liabilities),
            33 => Some(self.aggregate.total_shareholder_equity),
            34 => Some(self.aggregate.treasury_stock),
            35 => Some(self.aggregate.retained_earnings),
            36 => Some(self.aggregate.common_stock),
            37 => Some(self.aggregate.common_stock_shares_outstanding),
            _ => None,
        };
        self.index += 1;
        result
    }
}

impl Iterator for CashFlowIterator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let result = match self.index {
            0 => Some(self.aggregate.operating_cashflow),
            1 => Some(self.aggregate.payments_for_operating_activities),
            2 => Some(self.aggregate.proceeds_from_operating_activities),
            3 => Some(self.aggregate.change_in_operating_liabilities),
            4 => Some(self.aggregate.change_in_operating_assets),
            5 => Some(self.aggregate.depreciation_depletion_and_amortization),
            6 => Some(self.aggregate.capital_expenditures),
            7 => Some(self.aggregate.change_in_receivables),
            8 => Some(self.aggregate.change_in_inventory),
            9 => Some(self.aggregate.profit_loss),
            10 => Some(self.aggregate.cashflow_from_investment),
            11 => Some(self.aggregate.cashflow_from_financing),
            12 => Some(self.aggregate.proceeds_from_repayments_of_short_term_debt),
            13 => Some(self.aggregate.payments_for_repurchase_of_common_stock),
            14 => Some(self.aggregate.payments_for_repurchase_of_equity),
            15 => Some(self.aggregate.payments_for_repurchase_of_preferred_stock),
            16 => Some(self.aggregate.dividend_payout),
            17 => Some(self.aggregate.dividend_payout_common_stock),
            18 => Some(self.aggregate.dividend_payout_preferred_stock),
            19 => Some(self.aggregate.proceeds_from_issuance_of_common_stock),
            20 => Some(
                self.aggregate
                    .proceeds_from_issuance_of_long_term_debt_and_capital_securities_net,
            ),
            21 => Some(self.aggregate.proceeds_from_issuance_of_preferred_stock),
            22 => Some(self.aggregate.proceeds_from_repurchase_of_equity),
            23 => Some(self.aggregate.proceeds_from_sale_of_treasury_stock),
            24 => Some(self.aggregate.change_in_cash_and_cash_equivalents),
            25 => Some(self.aggregate.change_in_exchange_rate),
            26 => Some(self.aggregate.net_income),
            _ => None,
        };
        self.index += 1;
        result
    }
}

impl Iterator for EarningsIterator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let result = match self.index {
            0 => Some(self.aggregate.reported_eps),
            1 => Some(self.aggregate.estimated_eps),
            2 => Some(self.aggregate.surprise),
            _ => None,
        };
        self.index += 1;
        result
    }
}

impl Iterator for PriceAggregateIterator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let result = match self.index {
            0 => Some(self.aggregate.open),
            1 => Some(self.aggregate.high),
            2 => Some(self.aggregate.low),
            3 => Some(self.aggregate.close),
            4 => Some(self.aggregate.volume),
            _ => None,
        };
        self.index += 1;
        result
    }
}

impl Iterator for OptionsAggregateIterator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let result = match self.index {
            0 => Some(self.aggregate.last),
            1 => Some(self.aggregate.mark),
            2 => Some(self.aggregate.bid),
            4 => Some(self.aggregate.bid_size),
            5 => Some(self.aggregate.ask),
            6 => Some(self.aggregate.ask_size),
            7 => Some(self.aggregate.volume),
            8 => Some(self.aggregate.open_interest),
            9 => Some(self.aggregate.implied_volatility),
            10 => Some(self.aggregate.delta),
            11 => Some(self.aggregate.gamma),
            12 => Some(self.aggregate.theta),
            13 => Some(self.aggregate.vega),
            14 => Some(self.aggregate.rho),
            _ => None,
        };
        self.index += 1;
        result
    }
}

impl IntoIterator for IncomeStatement {
    type Item = f64;
    type IntoIter = IncomeStatementIterator;

    fn into_iter(self) -> Self::IntoIter {
        IncomeStatementIterator {
            aggregate: self,
            index: 0,
        }
    }
}

impl IntoIterator for BalanceSheet {
    type Item = f64;
    type IntoIter = BalanceSheetIterator;

    fn into_iter(self) -> Self::IntoIter {
        BalanceSheetIterator {
            aggregate: self,
            index: 0,
        }
    }
}

impl IntoIterator for CashFlow {
    type Item = f64;
    type IntoIter = CashFlowIterator;

    fn into_iter(self) -> Self::IntoIter {
        CashFlowIterator {
            aggregate: self,
            index: 0,
        }
    }
}

impl IntoIterator for Earnings {
    type Item = f64;
    type IntoIter = EarningsIterator;

    fn into_iter(self) -> Self::IntoIter {
        EarningsIterator {
            aggregate: self,
            index: 0,
        }
    }
}

impl IntoIterator for PriceAggregate {
    type Item = f64;
    type IntoIter = PriceAggregateIterator;

    fn into_iter(self) -> Self::IntoIter {
        PriceAggregateIterator {
            aggregate: self,
            index: 0,
        }
    }
}

impl IntoIterator for OptionsAggregate {
    type Item = f64;
    type IntoIter = OptionsAggregateIterator;

    fn into_iter(self) -> Self::IntoIter {
        OptionsAggregateIterator {
            aggregate: self,
            index: 0,
        }
    }
}

impl Aggregates for IncomeStatement {
    fn timestamp(&self) -> Instant {
        self.timestamp
    }

    fn duration(&self) -> Duration {
        self.duration
    }
}

impl Aggregates for BalanceSheet {
    fn timestamp(&self) -> Instant {
        self.timestamp
    }

    fn duration(&self) -> Duration {
        self.duration
    }
}

impl Aggregates for CashFlow {
    fn timestamp(&self) -> Instant {
        self.timestamp
    }

    fn duration(&self) -> Duration {
        self.duration
    }
}

impl Aggregates for Earnings {
    fn timestamp(&self) -> Instant {
        self.timestamp
    }

    fn duration(&self) -> Duration {
        self.duration
    }
}

impl Aggregates for PriceAggregate {
    fn timestamp(&self) -> Instant {
        self.timestamp
    }

    fn duration(&self) -> Duration {
        self.duration
    }
}

impl Aggregates for OptionsAggregate {
    fn timestamp(&self) -> Instant {
        self.timestamp
    }

    fn duration(&self) -> Duration {
        self.duration
    }
}
