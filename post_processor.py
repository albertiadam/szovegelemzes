"""
Post-processing of the LDA results.

Compares each company's dominant topic between the two years and summarizes the companies
that were skipped during text processing (TextProcessor).

Use-cases:
 - Build the topic transition (contingency) table between the previous and current year
 - Test whether the topics of the two years are related (chi-square test, Cramer's V)
 - Summarize the skipped companies by reason and by industry, and chart the counts

Generated outputs (in process_outputs/):
    - Contingency table of topic transitions (with totals): contingency_table.csv
    - Skipped companies grouped by industry and reason: grouped_skipped_companies.csv
    - Skipped company counts by reason and by industry: reason_counts.csv, industry_counts.csv
    - Chart of skipped companies by reason and top 10 industries: skipped_companies_counts.png
    - Printed to console: contingency table, number of unchanged companies, chi-square statistic,
      p-value, number of observations, Cramer's V and the skipped company summaries
"""
import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np
import matplotlib.pyplot as plt

class PostProcessor:
    """ PostProcessor class to summarize topic changes between the two years and the skipped companies. """
    def __init__(self, lda_result_string_path: str, skipped_data_path: str, years: list[str]) -> None:
        """
        Initialize the PostProcessor with the LDA results and the skipped companies files.

        :param lda_result_string_path: Path to the LDA comparison results CSV
            (company, previous_dominant_topic, current_dominant_topic).
        :param skipped_data_path: Path of the skipped file details CSVs, with a {year} placeholder.
        :param years: The two years compared. They are sorted, so the first one is the previous year.
        """
        self.data = pd.read_csv(lda_result_string_path)
        self.skipped_data_path = skipped_data_path
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

    def _get_skipped_companies_details(self) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Combine the skipped companies of both years and count them.
        A company skipped in both years is counted once, with its previous year reason and industry.

        :return: A tuple containing
            - skipped company counts grouped by industry and reason, sorted from most to least
            - skipped company counts per reason
            - skipped company counts per industry
        """
        skipped_data_prev = pd.read_csv(self.skipped_data_path.format(year=self.years[0]))
        skipped_data_curr = pd.read_csv(self.skipped_data_path.format(year=self.years[1]))

        all_skipped_companies = pd.concat([skipped_data_prev, skipped_data_curr]).drop_duplicates(subset=['company_name'])
        grouped_by = all_skipped_companies.groupby(['industry','reason']).agg({'company_name': 'count'}).reset_index().rename(columns={'company_name': 'skipped_companies_count'})
        reason_counts = all_skipped_companies['reason'].value_counts()
        industry_counts = all_skipped_companies['industry'].value_counts()
        return grouped_by.sort_values(by='skipped_companies_count', ascending=False), reason_counts, industry_counts


    def get_summary(self) -> None:
        """
        Run the post-processing: print the statistics of the topic changes and the skipped companies,
        save the tables and the chart to process_outputs/ and show the chart.

        Statistics of the topic changes:
            - chi-square test of independence on the transition table (without the totals)
            - Cramer's V: strength of the relationship between the topics of the two years,
              from 0 (no relationship) to 1 (topics fully determined by the previous year)
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

        grouped_by_skipped_companies, reason_counts, industry_counts = self._get_skipped_companies_details()
        print("[POST_PROCESS] Skipped Companies Details:")
        print(grouped_by_skipped_companies)
        grouped_by_skipped_companies.to_csv("process_outputs/grouped_skipped_companies.csv", index=False)
        print("[POST_PROCESS] Reason Counts:")
        print(reason_counts)
        reason_counts.to_csv("process_outputs/reason_counts.csv", header=True)
        print("[POST_PROCESS] Industry Counts:")
        print(industry_counts)
        sorted_industry_counts = industry_counts.sort_values(ascending=False)[0:10]
        industry_counts.to_csv("process_outputs/industry_counts.csv", header=True)

        fig, (ax_reason, ax_industry) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("Counts of skipped companies")

        ax_reason.bar(reason_counts.index, reason_counts.values)
        ax_reason.set_xlabel("Reason")
        ax_reason.set_ylabel("Count")
        plt.setp(ax_reason.get_xticklabels(), rotation=45, ha='right')

        industry_labels = [str(name).title() for name in sorted_industry_counts.index]
        ax_industry.bar(industry_labels, sorted_industry_counts.values)
        ax_industry.set_xlabel("Industry")
        ax_industry.set_ylabel("Count")
        plt.setp(ax_industry.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=9)

        fig.tight_layout()
        fig.savefig("process_outputs/skipped_companies_counts.png", bbox_inches="tight")
        plt.show()
