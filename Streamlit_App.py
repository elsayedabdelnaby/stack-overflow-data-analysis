import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page Configuration
st.set_page_config(page_title="Data Analysis App", layout="wide")

# Title
st.title("Stack Overflow Survey Analysis")

# Load Data
df = pd.read_csv("./survey_results_public_after_cleaning.csv")

# Sidebar Filters
st.sidebar.header("Filters")

# Multi-select for Developer Status
developer_status_options = df['Developer Status'].unique().tolist()
selected_developer_status = st.sidebar.multiselect(
    "Developer Status",
    options=developer_status_options,
    default=developer_status_options
)

# Multi-select for Remote Work
remote_work_options = df['RemoteWork'].unique().tolist()
selected_remote_work = st.sidebar.multiselect(
    "Remote Work",
    options=remote_work_options,
    default=remote_work_options
)

# Multi-select for Job
job_options = df['Job'].unique().tolist()
selected_job = st.sidebar.multiselect(
    "Job",
    options=job_options,
    default=job_options
)

# Multi-select for Country
country_options = df['Country'].unique().tolist()
selected_country = st.sidebar.multiselect(
    "Country",
    options=country_options,
    default=country_options
)

# Multi-select for Databases
database_options = ['PostgreSQL', 'MySQL', 'MongoDB', 'Microsoft SQL Server', 'SQLite']
selected_databases = st.sidebar.multiselect(
    "Databases",
    options=database_options,
    default=database_options
)

# Multi-select for Programming Languages
programming_language_options = ['Python', 'C#', 'Java', 'Rust', 'JavaScript', 'HTML/CSS', 'TypeScript']
selected_programming_languages = st.sidebar.multiselect(
    "Programming Languages",
    options=programming_language_options,
    default=programming_language_options
)

# Filter the dataframe based on selections
filtered_df = df[
    (df['Developer Status'].isin(selected_developer_status)) &
    (df['RemoteWork'].isin(selected_remote_work)) &
    (df['Job'].isin(selected_job)) &
    (df['Country'].isin(selected_country))
]

# Further filter by databases if needed
if selected_databases:
    # Filter rows based on selected databases
    db_filter = filtered_df[[f'DBAdmire_{db}' for db in selected_databases]].any(axis=1)
    filtered_df = filtered_df[db_filter]
    
    db_filter = filtered_df[[f'DBHaveWorkedWith_{db}' for db in selected_databases]].any(axis=1)
    filtered_df = filtered_df[db_filter]
    
    db_filter = filtered_df[[f'DBWantWorkedWith_{db}' for db in selected_databases]].any(axis=1)
    filtered_df = filtered_df[db_filter]

    # Drop columns for unselected databases in DBAdmire, DBHaveWorkedWith, and DBWantWorkedWith
    for db in database_options:
        if db not in selected_databases:
            filtered_df = filtered_df.drop(columns=[f'DBAdmire_{db}', f'DBHaveWorkedWith_{db}', f'DBWantWorkedWith_{db}'])

# Further filter by programming lanaguages if needed
if selected_programming_languages:
    # Filter rows based on selected programming lanaguages
    db_filter = filtered_df[[f'LanguageAdmired_{db}' for db in selected_programming_languages]].any(axis=1)
    filtered_df = filtered_df[db_filter]
    
    db_filter = filtered_df[[f'LanguageWantToWork_{db}' for db in selected_programming_languages]].any(axis=1)
    filtered_df = filtered_df[db_filter]
    
    db_filter = filtered_df[[f'HaveWorkedWith_{db}' for db in selected_programming_languages]].any(axis=1)
    filtered_df = filtered_df[db_filter]

    # Drop columns for unselected databases in LanguageAdmired, LanguageWantToWork, and HaveWorkedWith
    for db in programming_language_options:
        if db not in selected_programming_languages:
            filtered_df = filtered_df.drop(columns=[f'LanguageAdmired_{db}', f'LanguageWantToWork_{db}', f'HaveWorkedWith_{db}'])

# Tabs for Navigation
tabs = st.tabs([
    "Developer Status with Age", "EdLevel with Developer Status", "Job with Developer Status",
    "Remote Work", "Org Size with Remote Work", "Top 10 Countries", "Years to be Professional",
    "Coding Activities", "LearnCode", "Online Resources", "DB Admire", "DB Have Worked With",
    "DB Want Worked With", "Language Admire", "Language Want to Work", "Language Have Worked With",
    "Tech Docs"
])

# Developer Status with Age
with tabs[0]:
    st.subheader("Developer Status with Age")
    fig = px.histogram(data_frame=filtered_df, x=filtered_df['Age'], color=filtered_df['Developer Status'], barmode="group", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    fig = px.pie(data_frame=filtered_df, names=filtered_df['Age'])
    st.plotly_chart(fig, use_container_width=True)

# EdLevel with Developer Status
with tabs[1]:
    st.subheader("EdLevel with Developer Status")
    fig = px.pie(data_frame=filtered_df, names=filtered_df['EdLevel'])
    st.plotly_chart(fig, use_container_width=True)
    fig = px.histogram(data_frame=filtered_df, x=filtered_df['EdLevel'], color=filtered_df['Developer Status'], text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# Job with Developer Status
with tabs[2]:
    st.subheader("Job with Developer Status")
    fig = px.histogram(data_frame=filtered_df, y=filtered_df['Job'], text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    fig = px.histogram(data_frame=filtered_df, x=filtered_df['Developer Status'], color=filtered_df['Job'], text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# Remote Work
with tabs[3]:
    st.subheader("Remote Work")
    fig = px.pie(data_frame=filtered_df, names=filtered_df['RemoteWork'])
    st.plotly_chart(fig, use_container_width=True)
    fig = px.histogram(data_frame=filtered_df, x=filtered_df['Age'], color=filtered_df['RemoteWork'], barmode="group", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    fig = px.treemap(filtered_df[filtered_df['RemoteWork'] == 'Remote'], path=['Job'], title='Job Titles for Remote Workers (Treemap)')
    st.plotly_chart(fig, use_container_width=True)
    fig = px.treemap(filtered_df[filtered_df['RemoteWork'] == 'Hybrid'], path=['Job'], title='Job Titles for Hybrid Workers (Treemap)')
    st.plotly_chart(fig, use_container_width=True)

# Org Size with Remote Work
with tabs[4]:
    st.subheader("Org Size with Remote Work")
    fig = px.histogram(data_frame=filtered_df, x=filtered_df['OrgSize'], color=filtered_df['RemoteWork'], barmode="group", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    grouped = filtered_df.groupby(['OrgSize', 'RemoteWork']).size().unstack(fill_value=0)
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    percentages = percentages.reset_index().melt(id_vars='OrgSize', value_name='Percentage')
    fig = px.bar(percentages, y='Percentage', x='OrgSize', color='RemoteWork', barmode='group', text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# Top 10 Countries
with tabs[5]:
    st.subheader("Top 10 Countries")
    top_10_countries = filtered_df['Country'].value_counts().head(10).reset_index().rename(columns={'index': 'country', 'country': 'count'}).sort_values('count', ascending=False)
    fig = px.scatter(top_10_countries, y='count', x='Country', title='Top 10 Countries', labels={'count': 'Number of Respondents', 'country': 'Country'}, size='count', color='count')
    fig.update_traces(marker=dict(symbol='circle', line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
    fig.update_layout(showlegend=False, xaxis_title='Count', yaxis_title='Country')
    st.plotly_chart(fig, use_container_width=True)
    msk = (filtered_df['RemoteWork'] == 'Remote') | (filtered_df['RemoteWork'] == 'Hybrid')
    top_10_countries_remote_hybrid = filtered_df[msk]['Country'].value_counts().head(10).reset_index().rename(columns={'index': 'country', 'country': 'count'}).sort_values('count')
    fig = px.histogram(data_frame=top_10_countries_remote_hybrid, y=top_10_countries_remote_hybrid['Country'], x=top_10_countries_remote_hybrid['count'], text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# Years to be Professional
with tabs[6]:
    st.subheader("Years to be Professional")
    mean_value = (filtered_df['YearsCode'] - filtered_df['YearsCodePro']).mean()
    fig = go.Figure(go.Indicator(mode="number", value=mean_value, title={"text": f"Mean of Years to be Professional"}, number={'valueformat': '.2f'}))
    st.plotly_chart(fig, use_container_width=True)
    avg_years = filtered_df.groupby('EdLevel')[['YearsTobePro']].mean().reset_index()
    fig = px.scatter(avg_years, x='EdLevel', y='YearsTobePro', text='YearsTobePro', title='Average Years to Transition to Professional Coding by Education Level', labels={'YearsTobePro': 'Average Years', 'EdLevel': 'Education Level'})
    for i in range(len(avg_years)):
        fig.add_shape(type='line', x0=i, y0=0, x1=i, y1=avg_years.loc[i, 'YearsTobePro'], line=dict(color='gray', width=2))
    fig.update_traces(marker=dict(size=12, color='blue', line=dict(width=2, color='black')), texttemplate='%{y:.1f}', textposition='top center')
    fig.update_layout(xaxis=dict(tickvals=list(range(len(avg_years))), ticktext=avg_years['EdLevel']), yaxis_title='Average Years to Become Professional', template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    avg_years = filtered_df.groupby('Job')[['YearsTobePro']].mean().reset_index()
    fig = go.Figure(data=[go.Table(header=dict(values=['<b>Job</b>', '<b>Average Years to Become Proficient</b>'], fill_color='lightblue', align='center', font=dict(size=14, color='black')), cells=dict(values=[avg_years['Job'], avg_years['YearsTobePro']], fill_color='lightgrey', align='center', font=dict(size=12)))])
    fig.update_layout(title='Average Years to Become Proficient by Job', title_font=dict(size=20), margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

# Coding Activities
with tabs[7]:
    st.subheader("Coding Activities")
    st.dataframe(filtered_df[filtered_df['No Coding Outside Work'] == True][['Developer Status', 'Bootstrapping Business', 'Freelance or Contract Work']].value_counts())
    st.dataframe(filtered_df[filtered_df['No Coding Outside Work'] == False][['Developer Status', 'Bootstrapping Business', 'Freelance or Contract Work']].value_counts())

# LearnCode
with tabs[8]:
    st.subheader("LearnCode")
    learncode_columns = [col for col in filtered_df.columns if col.startswith('LearnCode_')]
    learncode_counts = filtered_df[learncode_columns].sum().reset_index()
    learncode_counts.columns = ['LearnCode', 'Count']
    fig = px.histogram(learncode_counts, x='LearnCode', y='Count', title='Count of True Values for Each LearnCode', labels={'Count': 'Number of True Values', 'LearnCode': 'LearnCode'}, color='LearnCode', text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    grouped = filtered_df.groupby('Developer Status')[learncode_columns].sum().reset_index()
    melted = grouped.melt(id_vars='Developer Status', var_name='LearnCode', value_name='Count')
    fig = px.scatter(melted, x='Developer Status', y='LearnCode', size='Count', color='Count', title='LearnCode Usage by Developer Status (Bubble Chart)', labels={'Count': 'Number of Trues', 'Developer Status': 'Developer Status', 'LearnCode': 'LearnCode'}, color_continuous_scale=px.colors.sequential.Plasma)
    st.plotly_chart(fig, use_container_width=True)
    grouped = filtered_df.groupby('Developer Status')[learncode_columns].sum()
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    percentages = percentages.reset_index()
    melted = percentages.melt(id_vars='Developer Status', var_name='LearnCode', value_name='Percentage')
    fig = px.histogram(melted, x='Developer Status', y='Percentage', color='LearnCode', barmode='group', title='Percentage of LearnCode Usage by Developer Status', labels={'Percentage': 'Percentage (%)', 'LearnCode': 'LearnCode', 'Developer Status': 'Developer Status'}, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    grouped = filtered_df.groupby('Age')[learncode_columns].sum()
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    percentages = percentages.reset_index()
    melted = percentages.melt(id_vars='Age', var_name='LearnCode', value_name='Percentage')
    fig = px.histogram(melted, x='Age', y='Percentage', color='LearnCode', barmode='group', title='Percentage of LearnCode Usage by Age', labels={'Percentage': 'Percentage (%)', 'LearnCode': 'LearnCode', 'Age': 'Age'}, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# Online Resources
with tabs[9]:
    st.subheader("Online Resources")
    online_resource_columns = [col for col in filtered_df.columns if col.startswith('OnlineResource_')]
    online_resource_counts = filtered_df[online_resource_columns].sum().reset_index()
    online_resource_counts.columns = ['OnlineResource', 'Count']
    online_resource_counts['OnlineResource'] = online_resource_counts['OnlineResource'].apply(lambda x: x.replace("OnlineResource_", ""))
    fig = px.histogram(online_resource_counts, x='Count', y='OnlineResource', title='Count of True Values for Each OnlineResource', labels={'Count': 'Number of True Values', 'OnlineResource': 'OnlineResource'}, color='OnlineResource', text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    grouped = filtered_df.groupby('Age')[online_resource_columns].sum()
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    percentages = percentages.reset_index()
    melted = percentages.melt(id_vars='Age', var_name='OnlineResource', value_name='Percentage')
    melted['OnlineResource'] = melted['OnlineResource'].apply(lambda x: x.replace("OnlineResource_", ""))
    fig = px.histogram(melted, x='Age', y='Percentage', color='OnlineResource', barmode='group', title='Percentage of OnlineResource Usage by Age', labels={'Percentage': 'Percentage (%)', 'OnlineResource': 'OnlineResource', 'Age': 'Age'}, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# DB Admire
with tabs[10]:
    st.subheader("DB Admire")
    db_admire_columns = [col for col in filtered_df.columns if col.startswith('DBAdmire_')]
    db_admire_counts = filtered_df[db_admire_columns].sum().reset_index()
    db_admire_counts.columns = ['DBAdmire', 'Count']
    db_admire_counts['DBAdmire'] = db_admire_counts['DBAdmire'].apply(lambda x: x.replace("DBAdmire_", ""))
    fig = px.histogram(db_admire_counts, x='DBAdmire', y='Count', title='Count of True Values for Each DBAdmire', labels={'Count': 'Number of True Values', 'DBAdmire': 'DBAdmire'}, color='DBAdmire', text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    grouped = filtered_df.groupby('Job')[db_admire_columns].sum()
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    percentages = percentages.reset_index()
    melted = percentages.melt(id_vars='Job', var_name='DBAdmire', value_name='Percentage')
    fig = px.histogram(melted, x='Percentage', y='Job', color='DBAdmire', barmode='group', title='Percentage of DBAdmire Usage by Job', labels={'Percentage': 'Percentage (%)', 'DBAdmire': 'DBAdmire', 'Job': 'Job'}, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# DB Have Worked With
with tabs[11]:
    st.subheader("DB Have Worked With")
    db_have_worked_columns = [col for col in filtered_df.columns if col.startswith('DBHaveWorkedWith_')]
    db_have_worked_counts = filtered_df[db_have_worked_columns].sum().reset_index()
    db_have_worked_counts.columns = ['DBHaveWorkedWith', 'Count']
    db_have_worked_counts['DBHaveWorkedWith'] = db_have_worked_counts['DBHaveWorkedWith'].apply(lambda x: x.replace("DBHaveWorkedWith_", ""))
    fig = px.histogram(db_have_worked_counts, x='DBHaveWorkedWith', y='Count', title='Count of True Values for Each DBHaveWorkedWith', labels={'Count': 'Number of True Values', 'DBHaveWorkedWith': 'DBHaveWorkedWith'}, color='DBHaveWorkedWith', text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    grouped = filtered_df.groupby('Developer Status')[db_have_worked_columns].sum()
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    percentages = percentages.reset_index()
    melted = percentages.melt(id_vars='Developer Status', var_name='DBHaveWorkedWith', value_name='Percentage')
    melted['DBHaveWorkedWith'] = melted['DBHaveWorkedWith'].apply(lambda x: x.replace("DBHaveWorkedWith_", ""))
    fig = px.histogram(melted, x='Developer Status', y='Percentage', color='DBHaveWorkedWith', barmode='group', title='Percentage of DBHaveWorkedWith Usage by Developer Status', labels={'Percentage': 'Percentage (%)', 'DBHaveWorkedWith': 'DBHaveWorkedWith', 'Developer Status': 'Developer Status'}, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# DB Want Worked With
with tabs[12]:
    st.subheader("DB Want Worked With")
    db_want_worked_columns = [col for col in filtered_df.columns if col.startswith('DBWantWorkedWith_')]
    db_want_worked_counts = filtered_df[db_want_worked_columns].sum().reset_index()
    db_want_worked_counts.columns = ['DBWantWorkedWith', 'Count']
    db_want_worked_counts['DBWantWorkedWith'] = db_want_worked_counts['DBWantWorkedWith'].apply(lambda x: x.replace("DBWantWorkedWith_", ""))
    fig = px.histogram(db_want_worked_counts, x='DBWantWorkedWith', y='Count', title='Count of True Values for Each DBWantWorkedWith', labels={'Count': 'Number of True Values', 'DBWantWorkedWith': 'DBWantWorkedWith'}, color='DBWantWorkedWith', text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    grouped = filtered_df.groupby('Job')[db_want_worked_columns].sum()
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    percentages = percentages.reset_index()
    melted = percentages.melt(id_vars='Job', var_name='DBWantWorkedWith', value_name='Percentage')
    fig = px.histogram(melted, x='Percentage', y='Job', color='DBWantWorkedWith', barmode='group', title='Percentage of DBWantWorkedWith Usage by Job', labels={'Percentage': 'Percentage (%)', 'DBWantWorkedWith': 'DBWantWorkedWith', 'Job': 'Job'}, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    
# Language Admire
with tabs[13]:
    st.subheader("Language Admire")
    language_admired_columns = [col for col in filtered_df.columns if col.startswith('LanguageAdmired_')]
    language_admired_counts = filtered_df[language_admired_columns].sum().reset_index()
    language_admired_counts.columns = ['LanguageAdmired', 'Count']
    language_admired_counts['LanguageAdmired'] = language_admired_counts['LanguageAdmired'].apply(lambda x: x.replace("LanguageAdmired_", ""))
    fig = px.histogram(language_admired_counts, x='LanguageAdmired', y='Count', title='Count of True Values for Each LanguageAdmired', labels={'Count': 'Number of True Values', 'LanguageAdmired': 'LanguageAdmired'}, color='LanguageAdmired', text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    grouped = filtered_df.groupby('Developer Status')[language_admired_columns].sum()
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    percentages = percentages.reset_index()
    melted = percentages.melt(id_vars='Developer Status', var_name='LanguageAdmired', value_name='Percentage')
    melted['LanguageAdmired'] = melted['LanguageAdmired'].apply(lambda x: x.replace("LanguageAdmired_", ""))
    fig = px.histogram(melted, x='Developer Status', y='Percentage', color='LanguageAdmired', barmode='group', title='Percentage of LanguageAdmired Usage by Developer Status', labels={'Percentage': 'Percentage (%)', 'LanguageAdmired': 'LanguageAdmired', 'Developer Status': 'Developer Status'}, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# Language Want to Work
with tabs[14]:
    st.subheader("Language Want to Work With")
    language_wantwork_columns = [col for col in filtered_df.columns if col.startswith('LanguageWantToWork_')]
    language_wantwork_counts = filtered_df[language_wantwork_columns].sum().reset_index()
    language_wantwork_counts.columns = ['LanguageWantToWork', 'Count']
    language_wantwork_counts['LanguageWantToWork'] = language_wantwork_counts['LanguageWantToWork'].apply(lambda x: x.replace("LanguageWantToWork_", ""))
    fig = px.histogram(language_wantwork_counts, x='LanguageWantToWork', y='Count', title='Count of True Values for Each LanguageWantToWork', labels={'Count': 'Number of True Values', 'LanguageWantToWork': 'LanguageWantToWork'}, color='LanguageWantToWork', text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    grouped = filtered_df.groupby('Developer Status')[language_wantwork_columns].sum()
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    percentages = percentages.reset_index()
    melted = percentages.melt(id_vars='Developer Status', var_name='LanguageWantToWork', value_name='Percentage')
    melted['LanguageWantToWork'] = melted['LanguageWantToWork'].apply(lambda x: x.replace("LanguageWantToWork_", ""))
    fig = px.histogram(melted, x='Developer Status', y='Percentage', color='LanguageWantToWork', barmode='group', title='Percentage of LanguageWantToWork Usage by Developer Status', labels={'Percentage': 'Percentage (%)', 'LanguageWantToWork': 'LanguageWantToWork', 'Developer Status': 'Developer Status'}, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# Language Have Worked With
with tabs[15]:
    st.subheader("Language Have Worked With")
    language_havework_columns = [col for col in filtered_df.columns if col.startswith('HaveWorkedWith_')]
    language_havework_counts = filtered_df[language_havework_columns].sum().reset_index()
    language_havework_counts.columns = ['HaveWorkedWith', 'Count']
    language_havework_counts['HaveWorkedWith'] = language_havework_counts['HaveWorkedWith'].apply(lambda x: x.replace("HaveWorkedWith_", ""))
    fig = px.histogram(language_havework_counts, x='HaveWorkedWith', y='Count', title='Count of True Values for Each HaveWorkedWith', labels={'Count': 'Number of True Values', 'HaveWorkedWith': 'HaveWorkedWith'}, color='HaveWorkedWith', text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    grouped = filtered_df.groupby('Developer Status')[language_havework_columns].sum()
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    percentages = percentages.reset_index()
    melted = percentages.melt(id_vars='Developer Status', var_name='HaveWorkedWith', value_name='Percentage')
    melted['HaveWorkedWith'] = melted['HaveWorkedWith'].apply(lambda x: x.replace("HaveWorkedWith_", ""))
    fig = px.histogram(melted, x='Developer Status', y='Percentage', color='HaveWorkedWith', barmode='group', title='Percentage of HaveWorkedWith Usage by Developer Status', labels={'Percentage': 'Percentage (%)', 'HaveWorkedWith': 'HaveWorkedWith', 'Developer Status': 'Developer Status'}, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# Tech Docs
with tabs[16]:
    st.subheader("Tech Docs")
    techdoc_columns = [col for col in filtered_df.columns if col.startswith('TechDoc_')]
    techdoc_counts = filtered_df[techdoc_columns].sum().reset_index()
    techdoc_counts.columns = ['TechDoc', 'Count']
    techdoc_counts['TechDoc'] = techdoc_counts['TechDoc'].apply(lambda x: x.replace("TechDoc_", ""))
    fig = px.histogram(
        techdoc_counts,
        x='TechDoc',
        y='Count',
        title='Count of True Values for Each TechDoc',
        labels={'Count': 'Number of True Values', 'TechDoc': 'TechDoc'},
        color='TechDoc',
        text_auto=True
    )
    st.plotly_chart(fig, use_container_width=True)
    grouped = filtered_df.groupby('Developer Status')[techdoc_columns].sum()
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    percentages = percentages.reset_index()
    melted = percentages.melt(id_vars='Developer Status', var_name='TechDoc', value_name='Percentage')
    melted['TechDoc'] = melted['TechDoc'].apply(lambda x: x.replace("TechDoc_", ""))
    fig = px.histogram(
        melted,
        x='Developer Status',
        y='Percentage',
        color='TechDoc',
        barmode='group',
        title='Percentage of TechDoc Usage by Developer Status',
        labels={'Percentage': 'Percentage (%)', 'TechDoc': 'TechDoc', 'Developer Status': 'Developer Status'},
        text_auto=True
    )
    st.plotly_chart(fig, use_container_width=True)
    grouped = filtered_df.groupby('Job')[techdoc_columns].sum()
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    percentages = percentages.reset_index()
    melted = percentages.melt(id_vars='Job', var_name='TechDoc', value_name='Percentage')
    melted['TechDoc'] = melted['TechDoc'].apply(lambda x: x.replace("TechDoc_", ""))

    pivot_table = melted.pivot(index='Job', columns='TechDoc', values='Percentage')

    header = ['Job'] + list(pivot_table.columns)
    values = [pivot_table.index] + [pivot_table[col].values for col in pivot_table.columns]

    fig = go.Figure(data=[
        go.Table(
            header=dict(
                values=[f"<b>{col}</b>" for col in header],
                fill_color="lightblue",
                align="center",
                font=dict(color="black", size=12),
            ),
            cells=dict(
                values=values,
                fill_color="white",
                align="center",
                format=[None] + [".2f"] * len(pivot_table.columns),  # Formatting percentages
                font=dict(color="black", size=11),
            )
        )
    ])

    fig.update_layout(
        title="Interactive Pivot Table: Percentage of TechDocs by Job",
        title_font=dict(size=18),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)