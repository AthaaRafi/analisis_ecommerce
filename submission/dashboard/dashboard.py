from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


st.set_page_config(page_title="Dashboard E-Commerce 2017-2018", layout="wide")


def apply_custom_style() -> None:
	st.markdown(
		"""
		<style>
		@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&display=swap');

		html, body, [class*="css"]  {
			font-family: 'DM Sans', sans-serif;
		}

		.stApp {
			background: linear-gradient(150deg, #f4f7f2 0%, #f3f6fb 45%, #f8f6ef 100%);
		}

		.block-container {
			padding-top: 1.5rem;
		}

		.metric-card {
			background: #ffffff;
			border: 1px solid #d9e3d6;
			border-left: 4px solid #0f766e;
			border-radius: 10px;
			padding: 14px 16px;
			box-shadow: 0 4px 18px rgba(15, 118, 110, 0.08);
		}

		.section-note {
			background: #fffaf1;
			border: 1px solid #f0d9a7;
			border-radius: 10px;
			padding: 12px 14px;
			color: #5d4a1f;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)


def safe_qcut(series: pd.Series, q: int, labels: list[int]) -> pd.Series:
	# Handle duplicated quantile edges by ranking values first.
	ranked = series.rank(method="first")
	return pd.qcut(ranked, q=q, labels=labels)


@st.cache_data
def load_and_prepare_data(base_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	data_dir = Path(base_path).resolve().parent.parent / "data"

	orders = pd.read_csv(data_dir / "orders_dataset.csv")
	order_items = pd.read_csv(data_dir / "order_items_dataset.csv")
	customers = pd.read_csv(data_dir / "customers_dataset.csv")

	merged_items = pd.merge(orders, order_items, on="order_id", how="inner")
	merged_items["order_purchase_timestamp"] = pd.to_datetime(
		merged_items["order_purchase_timestamp"]
	)

	merged_items = merged_items[
		(merged_items["order_purchase_timestamp"] >= "2017-01-01")
		& (merged_items["order_purchase_timestamp"] < "2019-01-01")
	].copy()

	merged_items["order_revenue"] = merged_items["price"] + merged_items["freight_value"]

	customer_orders = pd.merge(
		orders[["order_id", "customer_id", "order_purchase_timestamp"]],
		customers[["customer_id", "customer_unique_id"]],
		on="customer_id",
		how="inner",
	)
	customer_orders["order_purchase_timestamp"] = pd.to_datetime(
		customer_orders["order_purchase_timestamp"]
	)
	customer_orders = customer_orders[
		(customer_orders["order_purchase_timestamp"] >= "2017-01-01")
		& (customer_orders["order_purchase_timestamp"] < "2019-01-01")
	].copy()

	rfm_base = pd.merge(orders, order_items, on="order_id", how="inner")
	rfm_base = pd.merge(rfm_base, customers, on="customer_id", how="inner")
	rfm_base["order_purchase_timestamp"] = pd.to_datetime(rfm_base["order_purchase_timestamp"])
	rfm_base = rfm_base[
		(rfm_base["order_purchase_timestamp"] >= "2017-01-01")
		& (rfm_base["order_purchase_timestamp"] < "2019-01-01")
	].copy()

	return merged_items, customer_orders, rfm_base


def filter_by_date_range(
	merged_items: pd.DataFrame,
	customer_orders: pd.DataFrame,
	rfm_base: pd.DataFrame,
	start_date: pd.Timestamp,
	end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	merged_items_filtered = merged_items[
		merged_items["order_purchase_timestamp"].between(start_date, end_date)
	].copy()
	customer_orders_filtered = customer_orders[
		customer_orders["order_purchase_timestamp"].between(start_date, end_date)
	].copy()
	rfm_base_filtered = rfm_base[
		rfm_base["order_purchase_timestamp"].between(start_date, end_date)
	].copy()
	return merged_items_filtered, customer_orders_filtered, rfm_base_filtered


def build_monthly_metrics(merged_items: pd.DataFrame) -> pd.DataFrame:
	monthly = (
		merged_items.assign(Bulan=merged_items["order_purchase_timestamp"].dt.to_period("M"))
		.groupby("Bulan")
		.agg(GMV=("order_revenue", "sum"), Jumlah_Pesanan=("order_id", "nunique"))
		.reset_index()
	)
	monthly["AOV"] = monthly["GMV"] / monthly["Jumlah_Pesanan"]
	monthly["Bulan"] = monthly["Bulan"].dt.to_timestamp()
	return monthly


def build_loyal_metrics(
	merged_items: pd.DataFrame, customer_orders: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
	transaksi = (
		customer_orders.groupby("customer_unique_id")["order_id"]
		.nunique()
		.reset_index(name="total_transaksi")
	)
	loyal = transaksi[transaksi["total_transaksi"] > 3].copy()

	order_revenue = (
		merged_items.groupby("order_id")["order_revenue"].sum().reset_index(name="order_revenue")
	)
	oc_value = pd.merge(customer_orders, order_revenue, on="order_id", how="inner")

	loyal_ids = set(loyal["customer_unique_id"])
	total_customers = oc_value["customer_unique_id"].nunique()
	loyal_customers = float(loyal.shape[0])
	total_revenue = float(oc_value["order_revenue"].sum())
	loyal_revenue = float(
		oc_value[oc_value["customer_unique_id"].isin(loyal_ids)]["order_revenue"].sum()
	)
	loyal_revenue_pct = (loyal_revenue / total_revenue * 100) if total_revenue else 0.0

	summary = {
		"total_customers": float(total_customers),
		"loyal_customers": loyal_customers,
		"regular_customers": float(total_customers) - loyal_customers,
		"total_revenue": total_revenue,
		"loyal_revenue": loyal_revenue,
		"loyal_revenue_pct": loyal_revenue_pct,
	}
	return loyal, summary


def build_rfm_distribution(rfm_base: pd.DataFrame) -> pd.DataFrame:
	reference_date = rfm_base["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

	rfm = (
		rfm_base.groupby("customer_unique_id")
		.agg(
			recency=("order_purchase_timestamp", lambda x: (reference_date - x.max()).days),
			frequency=("order_id", "nunique"),
			monetary=("price", "sum"),
			freight=("freight_value", "sum"),
		)
		.reset_index()
	)
	rfm["monetary"] = rfm["monetary"] + rfm["freight"]
	rfm = rfm[["customer_unique_id", "recency", "frequency", "monetary"]]

	rfm["r_score"] = safe_qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
	rfm["f_score"] = pd.cut(
		rfm["frequency"], bins=[0, 1, 2, 3, 5, float("inf")], labels=[1, 2, 3, 4, 5]
	).astype(int)
	rfm["m_score"] = safe_qcut(rfm["monetary"], q=5, labels=[1, 2, 3, 4, 5]).astype(int)

	def segment(row: pd.Series) -> str:
		r_val = row["r_score"]
		f_val = row["f_score"]
		m_val = row["m_score"]
		if r_val >= 4 and f_val >= 4 and m_val >= 4:
			return "Champions"
		if r_val >= 3 and f_val >= 3:
			return "Loyal Customers"
		if r_val >= 3 and f_val <= 2 and m_val >= 4:
			return "Big Spenders"
		if r_val <= 2 and f_val >= 3:
			return "At Risk"
		if r_val <= 2 and f_val <= 2 and m_val <= 2:
			return "Hibernating"
		return "Others"

	rfm["segment"] = rfm.apply(segment, axis=1)
	dist = rfm["segment"].value_counts().reset_index()
	dist.columns = ["Segment", "Jumlah Pelanggan"]
	return dist


def render_header() -> None:
	st.markdown("## Dashboard Kinerja E-Commerce 2017-2018")
	st.markdown(
		"""
		<div class="section-note">
		Dashboard ini merangkum tren pesanan, GMV, AOV, proporsi pelanggan loyal (&gt;3 pesanan),
		dan segmentasi pelanggan berbasis RFM untuk mendukung keputusan retensi pelanggan.
		</div>
		""",
		unsafe_allow_html=True,
	)


def render_sidebar_filter(min_date: pd.Timestamp, max_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
	selected_range = st.sidebar.date_input(
		"Pilih rentang tanggal",
		value=(min_date.date(), max_date.date()),
		min_value=min_date.date(),
		max_value=max_date.date(),
	)
	if isinstance(selected_range, tuple):
		start_date, end_date = selected_range
	else:
		start_date = end_date = selected_range

	start_ts = pd.Timestamp(start_date)
	end_ts = pd.Timestamp(end_date)
	st.sidebar.caption(f"Menampilkan data {start_ts:%d %b %Y} s.d. {end_ts:%d %b %Y}")
	return start_ts, end_ts


def render_section_selector() -> str:
	return st.sidebar.selectbox(
		"Pilih tampilan",
		[
			"Tren Bulanan",
			"Pelanggan Loyal",
			"Segmentasi RFM",
			"Catatan Analisis",
		],
	)


def render_kpi_cards(monthly: pd.DataFrame, loyal_summary: dict[str, float]) -> None:
	total_orders = int(monthly["Jumlah_Pesanan"].sum())
	total_gmv = monthly["GMV"].sum()
	overall_aov = total_gmv / total_orders if total_orders else 0

	c1, c2, c3, c4 = st.columns(4)
	with c1:
		st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
		st.metric("Total Pesanan", f"{total_orders:,}")
		st.markdown("</div>", unsafe_allow_html=True)
	with c2:
		st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
		st.metric("Total GMV", f"BRL {total_gmv:,.0f}")
		st.markdown("</div>", unsafe_allow_html=True)
	with c3:
		st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
		st.metric("AOV Rata-rata", f"BRL {overall_aov:,.2f}")
		st.markdown("</div>", unsafe_allow_html=True)
	with c4:
		st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
		st.metric(
			"Loyal Revenue Share",
			f"{loyal_summary['loyal_revenue_pct']:.2f}%",
			help="Persentase pendapatan dari pelanggan dengan >3 pesanan.",
		)
		st.markdown("</div>", unsafe_allow_html=True)


def render_monthly_trend(monthly: pd.DataFrame) -> None:
	fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 16), sharex=True)

	sns.lineplot(data=monthly, x="Bulan", y="Jumlah_Pesanan", marker="o", color="#0f766e", ax=ax[0])
	ax[0].set_title("Tren Bulanan Jumlah Pesanan")
	ax[0].set_ylabel("Jumlah Pesanan")
	ax[0].grid(alpha=0.25)

	sns.lineplot(data=monthly, x="Bulan", y="GMV", marker="o", color="#ca8a04", ax=ax[1])
	ax[1].set_title("Tren Bulanan GMV")
	ax[1].set_ylabel("GMV (BRL)")
	ax[1].grid(alpha=0.25)

	sns.lineplot(data=monthly, x="Bulan", y="AOV", marker="o", color="#b91c1c", ax=ax[2])
	ax[2].set_title("Tren Bulanan AOV")
	ax[2].set_ylabel("AOV (BRL)")
	ax[2].set_xlabel("Bulan")
	ax[2].grid(alpha=0.25)

	plt.tight_layout()
	st.pyplot(fig)


def render_loyal_section(loyal_summary: dict[str, float]) -> None:
	col_left, col_right = st.columns(2)

	with col_left:
		count_df = pd.DataFrame(
			{
				"Kategori": ["Regular Customers", "Loyal Customers (>3)"] ,
				"Jumlah": [
					int(loyal_summary["regular_customers"]),
					int(loyal_summary["loyal_customers"]),
				],
			}
		)
		fig, ax = plt.subplots(figsize=(8, 5))
		sns.barplot(data=count_df, x="Kategori", y="Jumlah", palette=["#94a3b8", "#0f766e"], ax=ax)
		ax.set_title("Perbandingan Jumlah Pelanggan")
		ax.set_ylabel("Jumlah Pelanggan")
		ax.set_xlabel("")
		ax.tick_params(axis="x", rotation=8)
		for idx, value in enumerate(count_df["Jumlah"]):
			ax.text(idx, value, f"{value:,}", ha="center", va="bottom", fontsize=10)
		st.pyplot(fig)

	with col_right:
		rev_df = pd.DataFrame(
			{
				"Kategori": ["Regular Revenue", "Loyal Revenue"],
				"Persen": [100 - loyal_summary["loyal_revenue_pct"], loyal_summary["loyal_revenue_pct"]],
			}
		)
		fig, ax = plt.subplots(figsize=(8, 5))
		sns.barplot(data=rev_df, x="Kategori", y="Persen", palette=["#cbd5e1", "#b91c1c"], ax=ax)
		ax.set_title("Kontribusi Pendapatan per Kategori Pelanggan")
		ax.set_ylabel("Persentase Pendapatan (%)")
		ax.set_xlabel("")
		for idx, value in enumerate(rev_df["Persen"]):
			ax.text(idx, value, f"{value:.2f}%", ha="center", va="bottom", fontsize=10)
		st.pyplot(fig)


def render_rfm_segment(dist: pd.DataFrame) -> None:
	fig, ax = plt.subplots(figsize=(10, 5))
	dist = dist.sort_values("Jumlah Pelanggan", ascending=False)
	sns.barplot(data=dist, x="Segment", y="Jumlah Pelanggan", palette="crest", ax=ax)
	ax.set_title("Distribusi Segmen Pelanggan (RFM)")
	ax.set_xlabel("Segmen")
	ax.set_ylabel("Jumlah Pelanggan")
	ax.tick_params(axis="x", rotation=18)
	for idx, value in enumerate(dist["Jumlah Pelanggan"]):
		ax.text(idx, value, f"{int(value):,}", ha="center", va="bottom", fontsize=10)
	st.pyplot(fig)


def render_insight_notes(loyal_summary: dict[str, float]) -> None:
	st.markdown("### Ringkasan Insight")
	st.markdown(
		f"""
		- Tren pesanan dan GMV cenderung meningkat pada periode 2017-2018.
		- Nilai AOV berfluktuasi, menandakan frekuensi transaksi tidak selalu sejalan dengan nominal per order.
		- Pelanggan loyal (>3 pesanan) masih sedikit, dengan kontribusi pendapatan sekitar {loyal_summary['loyal_revenue_pct']:.2f}%.
		- Prioritas bisnis: perkuat repeat order melalui program retensi dan personalisasi kampanye untuk pelanggan bernilai tinggi.
		"""
	)


def main() -> None:
	apply_custom_style()
	render_header()

	merged_items, customer_orders, rfm_base = load_and_prepare_data(__file__)
	min_date = merged_items["order_purchase_timestamp"].min()
	max_date = merged_items["order_purchase_timestamp"].max()
	start_date, end_date = render_sidebar_filter(min_date, max_date)

	filtered_items, filtered_customer_orders, filtered_rfm_base = filter_by_date_range(
		merged_items, customer_orders, rfm_base, start_date, end_date
	)

	if filtered_items.empty:
		st.warning("Tidak ada data pada rentang tanggal yang dipilih. Silakan ubah filter.")
		return

	monthly = build_monthly_metrics(filtered_items)
	_loyal_df, loyal_summary = build_loyal_metrics(filtered_items, filtered_customer_orders)
	rfm_dist = build_rfm_distribution(filtered_rfm_base)

	render_kpi_cards(monthly, loyal_summary)

	selected_section = render_section_selector()

	if selected_section == "Tren Bulanan":
		render_monthly_trend(monthly)
	elif selected_section == "Pelanggan Loyal":
		render_loyal_section(loyal_summary)
	elif selected_section == "Segmentasi RFM":
		render_rfm_segment(rfm_dist)
	else:
		render_insight_notes(loyal_summary)


if __name__ == "__main__":
	main()