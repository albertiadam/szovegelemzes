"""
Post-processing of the LDA results.

Compares each company's dominant topic between the two years and summarizes the filings
that were skipped during text processing (TextProcessor).

Use-cases:
 - Build the topic transition (contingency) table between the previous and current year
 - Test whether the topics of the two years are related (chi-square test, Cramer's V)
 - Summarize the skipped filings by reason and by industry, and chart them (see _plot_skipped_companies)

Every skipped filing is counted, so a company skipped in both years is counted once in each year.
In the charts the rare reasons (item_1c_wrong_start, item_1c_too_long, missing_section_end, ...)
are merged into "other", otherwise their bars would be invisible.

Generated outputs (in process_outputs/):
    - Contingency table of topic transitions (with totals): contingency_table.csv
    - Skipped filings grouped by industry and reason: grouped_skipped_companies.csv
    - Skipped filing counts by reason and by industry: reason_counts.csv, industry_counts.csv
    - Figure 1, all skipped filings together: skipped_companies_counts.png
      left: counts per reason, right: top 10 industries, each bar stacked by reason
    - Figure 2, by year: skipped_companies_rates_by_year.png
      left: share of filings skipped per reason, right: share of filings skipped in the top 10 industries,
      2024 and 2025 side by side
    - Printed to console: contingency table, number of unchanged companies, chi-square statistic,
      p-value, number of observations, Cramer's V and the skipped filing summaries
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

class PostProcessor:
    """ PostProcessor class to summarize topic changes between the two years and the skipped companies. """
    def __init__(self, lda_result_string_path: str, skipped_data_path: str, filtered_data_path: str, years: list[str]) -> None:
        """
        Initialize the PostProcessor with the LDA results, the skipped filings and the filtered filings files.

        :param lda_result_string_path: Path to the LDA comparison results CSV
            (company, previous_dominant_topic, current_dominant_topic).
        :param skipped_data_path: Path of the skipped file details CSVs, with a {year} placeholder.
        :param filtered_data_path: Path of the filtered file details CSVs (every filing given to the
            TextProcessor, skipped or not), with a {year} placeholder. Used as the base of the skipped shares.
        :param years: The two years compared. They are sorted, so the first one is the previous year.
        """
        self.data = pd.read_csv(lda_result_string_path)
        self.skipped_data_path = skipped_data_path
        self.filtered_data_path = filtered_data_path
        self.years = sorted(years)

    def _get_cross_tab(self) -> pd.DataFrame:
        """
        Build the topic transition table: previous year topic (rows) against current year topic (columns).
        Companies without a dominant topic in either year are left out.
        Also prints the number of companies whose dominant topic did not change.

        :return: Crosstab of company counts, with an "All" row and column holding the totals.
        """
        cleaned_df = self.data.dropna(subset=['previous_dominant_topic', 'current_dominant_topic']).copy()
        cleaned_df['previous_dominant_topic'] = cleaned_df['previous_dominant_topic'].astype(int)
        cleaned_df['current_dominant_topic'] = cleaned_df['current_dominant_topic'].astype(int)

        unchanged = (
            cleaned_df['previous_dominant_topic'] == cleaned_df['current_dominant_topic']
        ).sum()
        print(f"[POST_PROCESS] Total unchanged companies: {unchanged}")
        trans_matrix = pd.crosstab(
            cleaned_df['previous_dominant_topic'],
            cleaned_df['current_dominant_topic'],
            rownames=[f'{self.years[0]}'],
            colnames=[f'{self.years[1]}'],
            margins=True
        )

        return trans_matrix

    def _get_skipped_companies_details(self) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
        """
        Combine the skipped filings of both years and count them. Also loads all filtered filings
        of both years (every filing given to the TextProcessor, skipped or not), the base of the skipped shares.
        Every skipped filing is counted, so a company skipped in both years is counted once in each year.
        Filings without an industry get the industry "Unknown".

        :return: A tuple containing
            - skipped filing counts grouped by industry and reason, sorted from most to least
            - skipped filing counts per reason
            - skipped filing counts per industry
            - all skipped filings of both years, with a year column
            - all filtered filings of both years, with a year column
        """
        skipped_data_prev = pd.read_csv(self.skipped_data_path.format(year=self.years[0]))
        skipped_data_curr = pd.read_csv(self.skipped_data_path.format(year=self.years[1]))
        skipped_data_prev['year'] = self.years[0]
        skipped_data_curr['year'] = self.years[1]
        filtered_data_prev = pd.read_csv(self.filtered_data_path.format(year=self.years[0]))
        filtered_data_curr = pd.read_csv(self.filtered_data_path.format(year=self.years[1]))
        filtered_data_prev['year'] = self.years[0]
        filtered_data_curr['year'] = self.years[1]

        all_skipped_companies = pd.concat([skipped_data_prev, skipped_data_curr])
        all_skipped_companies['industry'] = all_skipped_companies['industry'].fillna('Unknown')
        all_filings = pd.concat([filtered_data_prev, filtered_data_curr])
        all_filings['industry'] = all_filings['industry'].fillna('Unknown')
        grouped_by = all_skipped_companies.groupby(['industry','reason']).agg({'company_name': 'count'}).reset_index().rename(columns={'company_name': 'skipped_companies_count'})
        reason_counts = all_skipped_companies['reason'].value_counts()
        industry_counts = all_skipped_companies['industry'].value_counts()
        return grouped_by.sort_values(by='skipped_companies_count', ascending=False), reason_counts, industry_counts, all_skipped_companies, all_filings

    def _plot_skipped_companies(self, all_skipped_companies: pd.DataFrame, all_filings: pd.DataFrame) -> None:
        """
        Create the two figures of the skipped filings, save them to process_outputs/ and show them.
        The reasons other than missing_item_1c_cybersecurity and item_1c_too_short are merged into "other",
        otherwise their bars would be invisible.

        Figure 1, all skipped filings of both years together (skipped_companies_counts.png):
            - left: number of skipped filings per reason
            - right: the 10 industries with the most skipped filings, each bar stacked by reason
        Figure 2, by year (skipped_companies_rates_by_year.png):
            - left: share of all filings of the year skipped for each reason, 2024 and 2025 side by side
            - right: share of filings skipped in each of the same 10 industries, 2024 and 2025 side by side
            Shares are used instead of counts, because the number of filings differs between the years
            and between the industries.

        :param all_skipped_companies: All skipped filings of both years, with a year column
            (output of _get_skipped_companies_details).
        :param all_filings: All filtered filings of both years, with a year column
            (output of _get_skipped_companies_details).
        """
        skipped = all_skipped_companies.copy()
        main_reasons = ['missing_item_1c_cybersecurity', 'item_1c_too_short']
        skipped['reason'] = skipped['reason'].where(skipped['reason'].isin(main_reasons), 'other')
        top_industries = skipped['industry'].value_counts().index[:10]

        # Figure 1: counts of all skipped filings
        industry_by_reason = pd.crosstab(skipped['industry'], skipped['reason']).loc[top_industries]

        fig, (ax_reason, ax_industry) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("Counts of skipped filings")
        skipped['reason'].value_counts().plot.bar(ax=ax_reason, xlabel="Reason", ylabel="Count", rot=0)
        industry_by_reason.rename(index=str.title).plot.bar(ax=ax_industry, stacked=True, xlabel="Industry", ylabel="Count")
        plt.setp(ax_industry.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=9)
        fig.tight_layout()
        fig.savefig("process_outputs/skipped_companies_counts.png", bbox_inches="tight")

        # Figure 2: share of filings skipped, by year
        reason_rates = pd.crosstab(skipped['reason'], skipped['year']) / all_filings['year'].value_counts() * 100
        industry_rates = (
            pd.crosstab(skipped['industry'], skipped['year'])
            / pd.crosstab(all_filings['industry'], all_filings['year']) * 100
        ).loc[top_industries]

        year_colors = ['#9085e9', '#4b3bb5']  # previous year lighter, current year darker
        fig, (ax_reason, ax_industry) = plt.subplots(1, 2, figsize=(16, 7), layout='constrained')
        fig.suptitle("Share of skipped filings by year")
        reason_rates.plot.bar(ax=ax_reason, xlabel="Reason", ylabel="Share of all filings (%)", rot=0, color=year_colors)
        industry_rates.rename(index=str.title).plot.barh(ax=ax_industry, xlabel="Share of the industry's filings skipped (%)", ylabel="Industry", color=year_colors)
        ax_industry.invert_yaxis()  # most skipped industry at the top
        ax_industry.legend(title="year", loc='lower right')
        fig.savefig("process_outputs/skipped_companies_rates_by_year.png", bbox_inches="tight")
        plt.show()

    def get_summary(self) -> None:
        """
        Run the post-processing: print the statistics of the topic changes and the skipped filings,
        save the tables to process_outputs/, and create the charts of the skipped filings
        (see _plot_skipped_companies).

        Statistics of the topic changes:
            - chi-square test of independence on the transition table (without the totals)
            - Cramer's V: strength of the relationship between the topics of the two years,
              from 0 (no relationship) to 1 (topics fully determined by the previous year)

        Tables of the skipped filings (every reason kept separate):
            - grouped_skipped_companies.csv: counts by industry and reason
            - reason_counts.csv, industry_counts.csv: counts by reason and by industry
        """
        trans_matrix = self._get_cross_tab()
        print("[POST_PROCESS] Contingency Table:")
        print(trans_matrix)
        trans_matrix.to_csv("process_outputs/contingency_table.csv")

        trans_matrix = trans_matrix.drop(index="All", columns="All")

        chi2, p_value, _, _ = chi2_contingency(trans_matrix)
        print(f"[POST_PROCESS] Chi-square statistic: {chi2}")
        print(f"[POST_PROCESS] P-value: {p_value}")
        n = trans_matrix.sum().sum()
        r,k = trans_matrix.shape
        print(f"[POST_PROCESS] Total observations: {n}, Rows: {r}, Columns: {k}")
        cramers_v = np.sqrt(chi2 / (n * (min(r-1, k-1))))
        print(f"[POST_PROCESS] Cramer's V: {cramers_v}")

        grouped_by_skipped_companies, reason_counts, industry_counts, all_skipped_companies, all_filings = self._get_skipped_companies_details()
        print("[POST_PROCESS] Skipped Companies Details:")
        print(grouped_by_skipped_companies)
        grouped_by_skipped_companies.to_csv("process_outputs/grouped_skipped_companies.csv", index=False)
        print("[POST_PROCESS] Reason Counts:")
        print(reason_counts)
        reason_counts.to_csv("process_outputs/reason_counts.csv", header=True)
        print("[POST_PROCESS] Industry Counts:")
        print(industry_counts)
        industry_counts.to_csv("process_outputs/industry_counts.csv", header=True)

        self._plot_skipped_companies(all_skipped_companies, all_filings)
