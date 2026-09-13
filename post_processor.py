import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np

class PostProcessor:
    def __init__(self, lda_result_string_path:str,skipped_data_path:str,years:list[int]):
        self.data = pd.read_csv(lda_result_string_path)
        self.skipped_data_path = skipped_data_path
        self.years = sorted(years)

    def _get_cross_tab(self) -> pd.DataFrame:
        cleaned_df = self.data.dropna(subset=['previous_dominant_topic', 'current_dominant_topic']).copy()
        cleaned_df['previous_dominant_topic'] = cleaned_df['previous_dominant_topic'].astype(int)
        cleaned_df['current_dominant_topic'] = cleaned_df['current_dominant_topic'].astype(int)

        unchanged = (
            cleaned_df['previous_dominant_topic'] == cleaned_df['current_dominant_topic']
        ).sum()
        print(f"Total unchanged companies: {unchanged}")
        trans_matrix = pd.crosstab(
            cleaned_df['previous_dominant_topic'],
            cleaned_df['current_dominant_topic'],
            rownames=[f'{self.years[0]}'],
            colnames=[f'{self.years[1]}'],
            margins=True
        )

        return trans_matrix

    def _get_skipped_companies_details(self) -> pd.DataFrame:
        skipped_data_prev = pd.read_csv(self.skipped_data_path.format(year=self.years[0]))
        skipped_data_curr = pd.read_csv(self.skipped_data_path.format(year=self.years[1]))

        all_skipped_companies = pd.concat([skipped_data_prev, skipped_data_curr]).drop_duplicates(subset=['company_name'])
        grouped_by = all_skipped_companies.groupby(['industry','reason']).agg({'company_name': 'count'}).reset_index().rename(columns={'company_name': 'skipped_companies_count'})
        return grouped_by.sort_values(by='skipped_companies_count', ascending=False)


    def get_summary(self) -> None:
        trans_matrix = self._get_cross_tab()
        print("Contingency Table:")
        print(trans_matrix)
        trans_matrix.to_csv("process_outputs/contingency_table.csv")
        chi2, p_value, _, _ = chi2_contingency(trans_matrix)
        print(f"Chi-square statistic: {chi2}")
        print(f"P-value: {p_value}")
        n = trans_matrix.sum().sum()
        r,k = trans_matrix.shape
        cramers_v = np.sqrt(chi2 / (n * (min(r-1, k-1))))
        print(f"Cramer's V: {cramers_v}")

        grouped_by_skipped_companies = self._get_skipped_companies_details()
        print("Skipped Companies Details:")
        print(grouped_by_skipped_companies)
        grouped_by_skipped_companies.to_csv("process_outputs/grouped_skipped_companies.csv", index=False)
