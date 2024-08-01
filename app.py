import streamlit as st
import pandas as pd
import joypy
from joypy import joyplot
from plotly.graph_objs import Layout
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

st.sidebar.header('Main Sidebar')
main_option = st.sidebar.selectbox('Choose the page', ['Home', 'Truth Count Analysis', 'User Analysis', 'Engagement Analysis', 'New page'])

df_unique = pd.read_csv('influentialusers3.csv')
df_unique['created_at_iso'] = pd.to_datetime(df_unique['created_at_iso'])
df_unique['date'] = df_unique['created_at_iso'].dt.date
topics_to_plot = ['Judicial System', 'Trump', 'President Biden', 'Rally', 'Politics', 'Military']

if main_option == 'Home':
    st.title('Influential Truths')
    st.subheader("Data Description")
    st.write("This dataset captures engagement patterns for posts (called 'truths') from the top 30 most influential users on Truth Social over a month. It includes details on posts, interactions, and user information.")
    st.write("Purpose of dataset is to analyze engagement rates of users to see what kind of post generate the most engagement")
    st.write("Dataset can also be used to analyze viral truths or outliers by comparing the average truth's engagement pattern vs the outlier, exploring potential reasons behind it.")
    st.write(
        """
        We identified the top 30 most influential users on Truth Social by analyzing follower counts. These users served as the primary data sources for our dataset.
        """
    )
    st.subheader("Dataset Composition")
    st.write(
        """
        The dataset comprises truths posted by the identified influential users over a 100-hour period. Each truth was captured hourly, allowing us to observe engagement patterns over time.
        """
    )
    st.subheader("Data Collection and Analysis")
    st.write(
        """
        By collecting each truth 100 times, we can track changes in engagement metrics (replies, reblogs, favorites). This incremental analysis helps us understand:
        * How engagement evolves for a post after it's published
        * The types of posts that garner the most engagement
        """
    )

elif main_option == 'Truth Count Analysis':
    analysis_option = st.sidebar.selectbox('Choose the analysis type', ['Overall Frequency', 'Topic-specific Frequency'])

    if analysis_option == 'Overall Frequency':
        frequency_type = st.sidebar.selectbox('Choose frequency type', ['Daily Posting Frequency', 'Hourly Posting Frequency'])

        if frequency_type == 'Daily Posting Frequency':
            st.subheader('Daily Posting Frequency')
            st.write("Graph Description: Line plot showing the number of truths posted each day over time, to visualize the daily posting frequency of truths.")
            st.write("Potential Question: How does the number of truths posted vary day by day?")

            truths_per_day = df_unique.groupby('date').size().reset_index(name='truth_count')
            fig = px.line(truths_per_day, x='date', y='truth_count', title='Number of Truths Posted Per Day')
            fig.update_layout(width=1000, height=600)
            st.plotly_chart(fig)
        elif frequency_type == 'Hourly Posting Frequency':
            st.subheader('Hourly Posting Frequency')
            st.write("Graph Description: Line plot showing the number of truths posted each hour over time, to visualize the hourly posting frequency of truths.")
            st.write("Potential Question: How does the number of truths posted vary hour by hour?")

            df_unique['date_hour'] = df_unique['created_at_iso'].dt.floor('H')
            truths_per_hour = df_unique.groupby('date_hour').size().reset_index(name='truth_count')
            fig = px.line(truths_per_hour, x='date_hour', y='truth_count', title='Number of Truths Posted Per Hour')
            fig.update_layout(width=1000, height=600)
            st.plotly_chart(fig)

    elif analysis_option == 'Topic-specific Frequency':
        selected_topic = st.sidebar.selectbox('Choose a topic', topics_to_plot)
        frequency_type = st.sidebar.selectbox('Choose frequency type', ['Daily Posting Frequency', 'Hourly Posting Frequency'])
        filtered_df = df_unique[df_unique['topics'].str.contains(selected_topic, case=False, na=False)]

        if frequency_type == 'Daily Posting Frequency':
            st.subheader(f'Daily Posting Frequency for {selected_topic}')
            st.write(f"Graph Description: Line plot showing the number of truths posted each day for the topic '{selected_topic}'")
            st.write(f"Potential Question: How does the number of truths posted about '{selected_topic}' vary day by day?")

            # Overall daily posting frequency
            truths_per_day = filtered_df.groupby('date').size().reset_index(name='truth_count')
            fig = px.line(truths_per_day, x='date', y='truth_count', title=f'Number of Truths Posted Per Day for {selected_topic}')
            fig.update_layout(width=1000, height=600)
            st.plotly_chart(fig)

            # Daily posting frequency by user
            account_truths_per_day = filtered_df.groupby(['date', 'account_username']).size().reset_index(name='truth_count')
            fig = px.line(account_truths_per_day, 
                        x='date', 
                        y='truth_count', 
                        color='account_username',  
                        hover_data=['account_username', 'truth_count'],  
                        title=f'Number of Truths Posted Per Day by User for {selected_topic}')
            fig.update_layout(width=1000, height=600)
            st.plotly_chart(fig)

            # Box plot for top users
            user_totals = account_truths_per_day.groupby('account_username')['truth_count'].sum().sort_values(ascending=False)
            top_users = user_totals.head(10).index.tolist()  # Get top 10 users
            account_truths_per_day_top = account_truths_per_day[account_truths_per_day['account_username'].isin(top_users)]

            fig = px.box(account_truths_per_day_top, 
                        x='account_username', 
                        y='truth_count', 
                        color='account_username',
                        title=f'Distribution of Daily Truth Counts for Top Users - Topic: {selected_topic}',
                        labels={'truth_count': 'Daily Truth Count', 'account_username': 'Username'},
                        points="all")  # Show all points on the box plot
            fig.update_layout(
                xaxis_title='Username',
                yaxis_title='Daily Truth Count',
                showlegend=False,  
                xaxis={'categoryorder':'total descending'}  
            )
            fig.update_layout(width=1000, height=600)
            st.plotly_chart(fig)

            user_truth_counts = filtered_df.groupby('account_username')['id'].count().reset_index(name='truth_count')
            user_truth_counts = user_truth_counts.sort_values('truth_count', ascending=False)
            top_users = user_truth_counts.head(10)['account_username'].tolist()
            filtered_df_top = filtered_df[filtered_df['account_username'].isin(top_users)]
            fig, axes = joypy.joyplot(
                data=filtered_df_top,
                by='account_username',
                column='id',
                colormap=plt.cm.viridis,
                labels=top_users,
                range_style='all',
                tails=0.2,
                overlap=0.4,
                grid=True,
                title=f'Distribution of Truth Counts for Top Users - Topic: {selected_topic}'
            )
            plt.figure(figsize=(12, 8))
            plt.xlabel('Truth Count')
            plt.ylabel('Username')
            st.pyplot(fig)

        elif frequency_type == 'Hourly Posting Frequency':
            st.subheader(f'Hourly Posting Frequency for {selected_topic}')
            st.write(f"Graph Description: Line plot showing the number of truths posted each hour for the topic '{selected_topic}'.")
            st.write(f"Potential Question: How does the number of truths posted about '{selected_topic}' vary hour by hour?")

            filtered_df['date_hour'] = filtered_df['created_at_iso'].dt.floor('H')
            truths_per_hour = filtered_df.groupby('date_hour').size().reset_index(name='truth_count')
            fig = px.line(truths_per_hour, x='date_hour', y='truth_count', title=f'Number of Truths Posted Per Hour for {selected_topic}')
            fig.update_layout(width=800, height=600)
            st.plotly_chart(fig)
            
            account_truths_per_hour = filtered_df.groupby(['date_hour', 'account_username']).size().reset_index(name='truth_count')
            fig = px.line(account_truths_per_hour, 
                        x='date_hour', 
                        y='truth_count', 
                        color='account_username',  
                        hover_data=['account_username', 'truth_count'],  
                        title=f'Number of Truths Posted Per Hour for {selected_topic}')
            # fig.update_traces(
            #     hovertemplate="<br>".join([
            #         "Date and Hour: %{x}",
            #         "Username: %{customdata[0]}",
            #         "Truth Count: %{y}"
            #     ])
            # )
            # fig.update_layout(
            #     xaxis_title='Date and Hour',
            #     yaxis_title='Truth Count',
            #     legend_title='Username',
            #     hovermode='closest'  
            # )
            fig.update_layout(width=1000, height=600)
            st.plotly_chart(fig)

            account_truths_per_day = filtered_df.groupby([filtered_df['date_hour'].dt.date, 'account_username']).size().reset_index(name='truth_count')
            user_totals = account_truths_per_day.groupby('account_username')['truth_count'].sum().sort_values(ascending=False)
            top_users = user_totals.head(10).index.tolist()  # Get top 10 users
            account_truths_per_day_top = account_truths_per_day[account_truths_per_day['account_username'].isin(top_users)]

            figz = px.box(account_truths_per_day_top, 
                        x='account_username', 
                        y='truth_count', 
                        color='account_username',
                        title=f'Distribution of Daily Truth Counts for Top Users - Topic: {selected_topic}',
                        labels={'truth_count': 'Daily Truth Count', 'account_username': 'Username'},
                        points="all")  # Show all points on the box plot
            figz.update_layout(
                xaxis_title='Username',
                yaxis_title='Daily Truth Count',
                showlegend=False,  
                xaxis={'categoryorder':'total descending'}, 
                width=1000, height=600 
            )
            st.plotly_chart(figz)

            user_truth_counts = filtered_df.groupby('account_username')['id'].count().reset_index(name='truth_count')
            user_truth_counts = user_truth_counts.sort_values('truth_count', ascending=False)
            top_users = user_truth_counts.head(10)['account_username'].tolist()
            filtered_df_top = filtered_df[filtered_df['account_username'].isin(top_users)]
            fig, axes = joypy.joyplot(
                data=filtered_df_top,
                by='account_username',
                column='id',
                colormap=plt.cm.viridis,
                labels=top_users,
                range_style='all',
                tails=0.2,
                overlap=0.4,
                grid=True,
                title=f'Distribution of Truth Counts for Top Users - Topic: {selected_topic}'
            )
            plt.figure(figsize=(12, 8))
            plt.xlabel('Truth Count')
            plt.ylabel('Username')
            st.pyplot(fig)


elif main_option == 'User Analysis':
    st.subheader('User Analysis')
    analysis_option = st.sidebar.selectbox('Choose the analysis type', ['User Analysis', 'Confidence Intervals', 'Engagement Rate Analysis', 'Sentiment Analysis'])
    unique_usernames = ['None'] + list(df_unique['account_username'].unique())
    selected_username = st.sidebar.selectbox('Select a username', unique_usernames)
    all_topics = ['None'] + list(df_unique['topics'].explode().unique())
    selected_topic = st.sidebar.selectbox('Select a topic', all_topics)
    df_filtered = df_unique.copy()
    
    if selected_username != 'None':
        df_filtered = df_filtered[df_filtered['account_username'] == selected_username]
    if selected_topic != 'None':
        df_filtered = df_filtered[df_filtered['topics'].apply(lambda x: selected_topic in x if isinstance(x, list) else selected_topic == x)]
    if analysis_option == 'User Analysis':
        st.write("Graph Description: Line plot showing the follower count trend over time for the selected user.")
        st.write("Potential Question: How has the user's follower count changed over time?")
    
        if selected_username == 'None':
            st.warning("Please select a specific username to view follower count trend.")
        else:
            df_filtered['created_at_iso'] = pd.to_datetime(df_filtered['created_at_iso'])
            df_filtered = df_filtered.sort_values('created_at_iso')
            fig_followers = go.Figure()
            fig_followers.add_trace(go.Scatter(
                x=df_filtered['created_at_iso'], 
                y=df_filtered['account_followers_count'], 
                mode='lines+markers',
                name='Follower Count'
            ))
            fig_followers.update_layout(
                title=f'Follower Count Trend for {selected_username}',
                xaxis_title='Date',
                yaxis_title='Follower Count',
                template='plotly_white',
                width=1000,
                height=600
            )
            st.plotly_chart(fig_followers)

        st.write("Graph Description: Bar chart showing the frequency of the most common topics in the dataset.")
        st.write("Potential Question: What are the most common topics discussed in the dataset?")

        all_topics = [topic for topics in df_filtered['topics'] for topic in (topics.keys() if isinstance(topics, dict) else [topics])]
        topic_counter = Counter(all_topics)
        topic_df = pd.DataFrame(topic_counter.items(), columns=['Topic', 'Frequency']).sort_values(by='Frequency', ascending=False)
        fig = go.Figure(go.Bar(x=topic_df['Topic'], y=topic_df['Frequency'], marker=dict(color='skyblue')))
        title_suffix = f" for {selected_username}" if selected_username != 'None' else ""
        fig.update_layout(
            title=f'Most Common Topics in the Dataset{title_suffix}',
            xaxis_title='Topics',
            yaxis_title='Frequency',
            xaxis_tickangle=-45,
            template='plotly_white',
            width=1000,
            height=600
        )
        st.plotly_chart(fig)

        st.write("Graph Description: Line plot showing the daily count of truths over time for selected topics.")
        st.write("Potential Question: How does the daily posting frequency of truths vary over time for selected topics?")
        fig = go.Figure()
        for topic in set(all_topics):
            filtered_topic_df = df_filtered[df_filtered['topics'].apply(lambda x: topic in x if isinstance(x, list) else topic == x)]
            daily_topic_count = filtered_topic_df.groupby('date')['created_at_iso'].count().reset_index(name='truth_count')
            fig.add_trace(go.Scatter(x=daily_topic_count['date'], y=daily_topic_count['truth_count'], mode='lines', name=topic))
        fig.update_layout(
            title=f'Truth Counts Over Time for Selected Topics{title_suffix}',
            xaxis_title='Date',
            yaxis_title='Truth Count',
            legend_title='Topic',
            width=1000,
            height=600
        )
        st.plotly_chart(fig)

        st.write("Graph Description: Stacked bar chart showing the distribution of positive, negative, and neutral sentiments across topics for a specific user.")
        st.write("Potential Question: How does the sentiment distribution across different topics look for the selected user?")
        if selected_username == 'None':
            st.warning("Please select a specific username to view sentiment distribution.")
        else:
            user_topics = df_filtered.groupby(['sentiment_label'])['topics'].value_counts().unstack(fill_value=0)
            sentiment_counts = pd.DataFrame({
                'Topic': user_topics.columns,
                'Positive': user_topics.loc['positive'] if 'positive' in user_topics.index else pd.Series(0, index=user_topics.columns),
                'Negative': user_topics.loc['negative'] if 'negative' in user_topics.index else pd.Series(0, index=user_topics.columns),
                'Neutral': user_topics.loc['neutral'] if 'neutral' in user_topics.index else pd.Series(0, index=user_topics.columns),
            }).sort_values(by=['Positive', 'Negative', 'Neutral'], ascending=False)
            fig = go.Figure()
            for sentiment, color in zip(['Positive', 'Negative', 'Neutral'], ['green', 'red', 'gray']):
                fig.add_trace(go.Bar(x=sentiment_counts['Topic'], y=sentiment_counts[sentiment], name=sentiment, marker=dict(color=color)))
            fig.update_layout(
                title=f"Sentiment Distribution for Most Common Topics of {selected_username}",
                xaxis_title='Topics',
                yaxis_title='Count',
                xaxis_tickangle=-45,
                template='plotly_white',
                legend_title="Sentiment",
                legend=dict(x=1, xanchor='right'),
                width=1000,
                height=600
            )
            st.plotly_chart(fig)

    elif analysis_option == 'Confidence Intervals':
        st.write("Graph Description: Scatter plots with confidence intervals displaying the mean values of different metrics for each username.")
        st.write("Purpose: To visualize the average performance and variability of different engagement metrics across usernames with 95% confidence intervals.")
        st.write("Potential Question: How do the mean values and confidence intervals of various engagement metrics differ across usernames?")
        columns_to_plot = ['engagement_rate', 'avg_engagement', 'favourites_count', 'replies_count', 'reblogs_count', 'sentiment_score']
        selected_metric = st.selectbox('Select a metric to plot', columns_to_plot)
        if selected_username != 'None':
            usernames_to_plot = [selected_username]
        else:
            usernames_to_plot = df_unique['account_username'].unique()
        means = df_unique[df_unique['account_username'].isin(usernames_to_plot)].groupby('account_username')[selected_metric].mean()
        stds = df_unique[df_unique['account_username'].isin(usernames_to_plot)].groupby('account_username')[selected_metric].std()
        counts = df_unique[df_unique['account_username'].isin(usernames_to_plot)].groupby('account_username')[selected_metric].count()
        standard_errors = stds / np.sqrt(counts)
        confidence_intervals = stats.t.interval(0.95, counts - 1, loc=means, scale=standard_errors)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=means.index,
            y=means,
            mode='markers',
            name=f'Mean {selected_metric}',
            error_y=dict(
                type='data',
                array=(confidence_intervals[1] - means),
                arrayminus=(means - confidence_intervals[0]),
                visible=True,
                color='rgba(0,0,0,0.6)'
            )
        ))
        fig.add_trace(go.Scatter(x=means.index, y=confidence_intervals[0], mode='lines', name='Lower CI'))
        fig.add_trace(go.Scatter(x=means.index, y=confidence_intervals[1], mode='lines', name='Upper CI'))
        fig.update_layout(
            title=f'Mean {selected_metric} and 95% Confidence Intervals',
            xaxis_title='Username',
            yaxis_title=selected_metric,
            xaxis={'tickangle': 45},
            showlegend=True,
            hovermode='x unified',
            width=1000,
            height=600
        )
        st.plotly_chart(fig)

    elif analysis_option == 'Engagement Rate Analysis':
        st.write("Graph Description: Engagement Rate of Users Over Time (Unique Truths)")
        st.write("Purpose: To visualize the engagement rate trends of unique tweets by realDonaldTrump and compare individual tweet engagement rates to the overall average.")
        st.write("Potential Question: What are the trends in engagement rate for realDonaldTrump's unique tweets, and how does the engagement rate of individual tweets compare to the average?")
        df_filtered['collection_time'] = pd.to_datetime(df_filtered['collection_time'])
        df_latest = df_filtered.loc[df_filtered.groupby('id')['collection_time'].idxmax()].drop_duplicates(subset='id').sort_values(by='collection_time')
        df_latest.set_index('collection_time', inplace=True)
        df_latest['engagement_rate'] = pd.to_numeric(df_latest['engagement_rate'], errors='coerce')
        fig = go.Figure(go.Scatter(x=df_latest.index, y=df_latest['engagement_rate'], mode='lines+markers', name='Engagement Rate'))
        average_engagement_rate = df_latest['engagement_rate'].mean()
        fig.add_trace(go.Scatter(x=df_latest.index, y=[average_engagement_rate]*len(df_latest), mode='lines', name='Average Engagement Rate', line=dict(dash='dash', color='red')))
        fig.update_layout(
            title=f'Engagement Rate of {selected_username} Over Time (Unique Truths)',
            xaxis_title='Date',
            yaxis_title='Engagement Rate',
            legend_title='Legend',
            xaxis_tickangle=-45,
            template='plotly_white',
            width=1000,
            height=600
        )
        st.plotly_chart(fig)
        max_engagement_rate_row = df_latest.loc[df_latest['engagement_rate'].idxmax()]
        if isinstance(max_engagement_rate_row, pd.DataFrame):
            max_engagement_rate_row = max_engagement_rate_row.iloc[0]
        st.write(f'The highest engagement rate value is: {max_engagement_rate_row["engagement_rate"]}')
        st.write(f'The corresponding id is: {max_engagement_rate_row["id"]}')
        st.write(f'Number of unique tweets: {df_latest.shape[0]}')

    elif analysis_option == 'Sentiment Analysis':
        st.write("Graph Description: Daily and Hourly Sentiment Analysis, and Sentiment Distribution Across Topics")
        st.write("""
            Potential Questions: 
            1. How do the positive, negative, and neutral sentiments of the user's posts change over time (daily and hourly)?
            2. How does the sentiment distribution across different topics look for the selected user?
        """)
        if selected_username == 'None':
            st.warning("Please select a specific username for sentiment analysis.")
        else:
            df_filtered['created_at_iso'] = pd.to_datetime(df_filtered['created_at_iso'])
            df_filtered['sentiment_label'] = df_filtered['sentiment_label'].str.lower()
            color_map = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
            # Daily Sentiment Analysis
            df_filtered['date'] = df_filtered['created_at_iso'].dt.date
            daily_sentiment = df_filtered.groupby(['date', 'sentiment_label']).size().unstack(fill_value=0)
            daily_sentiment = daily_sentiment.div(daily_sentiment.sum(axis=1), axis=0)
            fig_daily = go.Figure()
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment in daily_sentiment.columns:
                    fig_daily.add_trace(go.Scatter(
                        x=daily_sentiment.index, 
                        y=daily_sentiment[sentiment], 
                        mode='lines+markers', 
                        name=sentiment.capitalize(),
                        line=dict(color=color_map[sentiment])
                    ))
            fig_daily.update_layout(
                title=f'Daily Sentiment Analysis for {selected_username}',
                xaxis_title='Date',
                yaxis_title='Proportion of Sentiments',
                template='plotly_white',
                width=1000,
                height=600
            )
            st.plotly_chart(fig_daily)

            # Hourly Sentiment Analysis
            df_filtered['hour'] = df_filtered['created_at_iso'].dt.hour
            hourly_sentiment = df_filtered.groupby(['hour', 'sentiment_label']).size().unstack(fill_value=0)
            hourly_sentiment = hourly_sentiment.div(hourly_sentiment.sum(axis=1), axis=0)
            fig_hourly = go.Figure()
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment in hourly_sentiment.columns:
                    fig_hourly.add_trace(go.Scatter(
                        x=hourly_sentiment.index, 
                        y=hourly_sentiment[sentiment], 
                        mode='lines+markers', 
                        name=sentiment.capitalize(),
                        line=dict(color=color_map[sentiment])
                    ))
            fig_hourly.update_layout(
                title=f'Hourly Sentiment Analysis for {selected_username}',
                xaxis_title='Hour of Day',
                yaxis_title='Proportion of Sentiments',
                template='plotly_white',
                width=1000,
                height=600
            )
            st.plotly_chart(fig_hourly)

            # Sentiment Distribution Across Topics
            user_topics = df_filtered.groupby(['sentiment_label'])['topics'].value_counts().unstack(fill_value=0)
            sentiment_counts = pd.DataFrame({
                'Topic': user_topics.columns,
                'Positive': user_topics.loc['positive'] if 'positive' in user_topics.index else pd.Series(0, index=user_topics.columns),
                'Negative': user_topics.loc['negative'] if 'negative' in user_topics.index else pd.Series(0, index=user_topics.columns),
                'Neutral': user_topics.loc['neutral'] if 'neutral' in user_topics.index else pd.Series(0, index=user_topics.columns),
            }).sort_values(by=['Positive', 'Negative', 'Neutral'], ascending=False)
            fig_topics = go.Figure()
            for sentiment in ['Positive', 'Negative', 'Neutral']:
                fig_topics.add_trace(go.Bar(
                    x=sentiment_counts['Topic'], 
                    y=sentiment_counts[sentiment], 
                    name=sentiment, 
                    marker=dict(color=color_map[sentiment.lower()])
                ))
            fig_topics.update_layout(
                title=f"Sentiment Distribution for Most Common Topics of {selected_username}",
                xaxis_title='Topics',
                yaxis_title='Count',
                xaxis_tickangle=-45,
                template='plotly_white',
                legend_title="Sentiment",
                legend=dict(x=1, xanchor='right'),
                width=1000,
                height=600
            )
            st.plotly_chart(fig_topics)

            # Additional statistics
            total_posts = len(df_filtered)
            sentiment_counts = df_filtered['sentiment_label'].value_counts()
            st.write(f"Total posts: {total_posts}")
            st.write(f"Positive posts: {sentiment_counts.get('positive', 0)} ({sentiment_counts.get('positive', 0)/total_posts:.2%})")
            st.write(f"Negative posts: {sentiment_counts.get('negative', 0)} ({sentiment_counts.get('negative', 0)/total_posts:.2%})")
            st.write(f"Neutral posts: {sentiment_counts.get('neutral', 0)} ({sentiment_counts.get('neutral', 0)/total_posts:.2%})")
            st.write(f"Day with highest positive sentiment: {daily_sentiment['positive'].idxmax() if 'positive' in daily_sentiment.columns else 'N/A'}")
            st.write(f"Day with highest negative sentiment: {daily_sentiment['negative'].idxmax() if 'negative' in daily_sentiment.columns else 'N/A'}")
            st.write(f"Hour with highest positive sentiment: {hourly_sentiment['positive'].idxmax() if 'positive' in hourly_sentiment.columns else 'N/A'}:00")
            st.write(f"Hour with highest negative sentiment: {hourly_sentiment['negative'].idxmax() if 'negative' in hourly_sentiment.columns else 'N/A'}:00")
           

elif main_option == 'Engagement Analysis':
    st.subheader('Engagement Analysis')
    st.write("Purpose: To visualize engagement metrics for specific users, topics, or both.")

    # Get unique topics and usernames
    unique_topics = df_unique['topics'].explode().unique()
    unique_usernames = df_unique['account_username'].unique()

    st.sidebar.subheader("Analysis Type")
    analysis_type = st.sidebar.radio("Select analysis type", ["Individual Analysis", "Combined Analysis"])

    if analysis_type == "Individual Analysis":
        st.sidebar.subheader("Plot Type")
        plot_type = st.sidebar.selectbox("Select plot type", ["Scatter Plot", "Box Plot", "Ridge Line Plot", "Confidence Intervals"])
        
        st.sidebar.subheader("Topic Selection")
        selected_topics = st.sidebar.multiselect('Select topics', list(unique_topics), default=list(unique_topics)[:5])
        
        st.sidebar.subheader("Metric Selection")
        columns_to_plot = ['engagement_rate', 'avg_engagement', 'favourites_count', 'replies_count', 'reblogs_count', 'sentiment_score']
        selected_columns = st.sidebar.multiselect('Select metrics to plot', columns_to_plot, default=['engagement_rate'])

        if not selected_topics:
            st.warning("Please select at least one topic.")
        elif not selected_columns:
            st.warning("Please select at least one engagement metric to plot.")
        else:
            for col in selected_columns:
                if plot_type == "Scatter Plot":
                    fig = go.Figure()
                    for topic in selected_topics:
                        topic_data = df_unique[df_unique['topics'].apply(lambda x: topic in x if isinstance(x, list) else topic == x)]
                        if not topic_data.empty:
                            normalized_data = (topic_data[col] - topic_data[col].min()) / (topic_data[col].max() - topic_data[col].min())
                            trace = go.Scatter(
                                x=topic_data['created_at_iso'],
                                y=normalized_data,
                                mode='markers',
                                name=topic,
                            )
                            fig.add_trace(trace)
                    topics_str = ", ".join(selected_topics)            
                    layout = go.Layout(
                        title=f"Engagement Metrics for {topics_str} (Normalized - {col})",
                        xaxis={"title": "Created At (ISO)"},
                        yaxis={"title": col},
                    )
                    fig.update_layout(layout)
                    fig.update_layout(width=1000, height=600)
                    st.plotly_chart(fig)

                    fig2 = go.Figure()
                    for topic in selected_topics:
                        topic_data = df_unique[df_unique['topics'].apply(lambda x: topic in x if isinstance(x, list) else topic == x)]
                        if not topic_data.empty:
                            trace = go.Scatter(
                                x=topic_data['created_at_iso'],
                                y=topic_data[col],
                                mode='markers',
                                name=topic,
                            )
                            fig2.add_trace(trace)
                    fig2.update_layout(
                        title=f"{col} by {topics_str}",
                        xaxis_title="Created At (ISO)",
                        yaxis_title=col,
                        yaxis_type="log"  
                    )
                    fig2.update_layout(width=1000, height=600)
                    st.plotly_chart(fig2)

                elif plot_type == "Box Plot":
                    st.write("Graph Description: Box plots showing the distribution of engagement metrics for selected topics.")
                    combined_topic_data = pd.DataFrame()
                    for topic in selected_topics:
                        topic_data = df_unique[df_unique['topics'].apply(lambda x: topic in x if isinstance(x, list) else topic == x)]
                        combined_topic_data = pd.concat([combined_topic_data, topic_data])
                    fig = px.box(combined_topic_data, 
                                x='topics', 
                                y=col, 
                                color='topics',
                                title=f'Distribution of {col} for Selected Topics',
                                labels={col: col.capitalize(), 'topics': 'Topic'},
                                points="all")
                    fig.update_layout(
                        xaxis_title='Topic',
                        yaxis_title=col.capitalize(),
                        showlegend=True,  
                        xaxis={'categoryorder':'total descending'},
                        width=1000,
                        height=600
                    )
                    st.plotly_chart(fig)

                elif plot_type == "Ridge Line Plot":
                    st.write("Graph Description: Ridge line plots showing the distribution of engagement metrics for selected topics.")
                    fig = go.Figure()
                    y_offset = 0
                    colors = px.colors.qualitative.Plotly  
                    for i, topic in enumerate(selected_topics):
                        topic_data = df_unique[df_unique['topics'].apply(lambda x: topic in x if isinstance(x, list) else topic == x)]
                        if not topic_data.empty:
                            kde = stats.gaussian_kde(topic_data[col])
                            x_range = np.linspace(topic_data[col].min(), topic_data[col].max(), 100)
                            y_kde = kde(x_range)
                            fig.add_trace(go.Scatter(
                                x=x_range,
                                y=y_kde + y_offset,
                                fill='tozeroy',
                                name=topic,
                                line=dict(color=colors[i % len(colors)])  
                            ))
                            y_offset += np.max(y_kde)
                    fig.update_layout(
                        title=f"Ridge Line Plot of {col} for Selected Topics",
                        xaxis_title=col.capitalize(),
                        yaxis_title="Density",
                        showlegend=True,
                        width=1000,
                        height=600
                    )
                    st.plotly_chart(fig)
                    
                    st.write("Graph Description: Ridge line plots showing the distribution of engagement metrics for selected topics.")
                    filtered_df = df_unique[df_unique['topics'].isin(selected_topics)]
                    fig, axes = joyplot(
                        data=filtered_df,
                        by='topics',
                        column='engagement_rate',
                        colormap=plt.cm.viridis,
                        labels=selected_topics,
                        range_style='all',
                        tails=0.2,
                        overlap=0.4,
                        grid=True
                    )
                    plt.title(f'Distribution of Engagement Rate for Selected Topics')
                    plt.xlabel('Engagement Rate')
                    plt.ylabel('Topics')
                    st.pyplot(plt.gcf())
                
                elif plot_type == "Confidence Intervals":
                    st.write("Confidence Intervals")
                    st.write("Graph Description: Scatter plots with confidence intervals displaying the mean values of different metrics for each topic.")
                    st.write("Purpose: To visualize the average performance and variability of different engagement metrics across topics with 95% confidence intervals.")
                    st.write("Potential Question: How do the mean values and confidence intervals of various engagement metrics differ across topics?")

                    filtered_df = df_unique[df_unique['topics'].isin(selected_topics)]
                    
                    means = filtered_df.groupby('topics')[col].mean()
                    stds = filtered_df.groupby('topics')[col].std()
                    counts = filtered_df.groupby('topics')[col].count()

                    standard_errors = stds / np.sqrt(counts)
                    confidence_intervals = stats.t.interval(0.95, counts - 1, loc=means, scale=standard_errors)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=means.index,
                        y=means,
                        mode='markers',
                        name='Mean ' + col,
                        error_y=dict(
                            type='data',
                            array=(confidence_intervals[1] - means),
                            arrayminus=(means - confidence_intervals[0]),
                            visible=True,
                            color='rgba(0,0,0,0.6)'
                        )
                    ))
                    fig.add_trace(go.Scatter(x=means.index, y=confidence_intervals[0], mode='lines', name='Lower CI'))
                    fig.add_trace(go.Scatter(x=means.index, y=confidence_intervals[1], mode='lines', name='Upper CI'))
                    fig.update_layout(
                        title=f'Mean {col} and 95% Confidence Intervals',
                        xaxis_title='Topics',
                        yaxis_title=col,
                        xaxis={'tickangle': 45},
                        showlegend=True,
                        hovermode='x unified',
                        width=1000,
                        height=600
                    )
                    st.plotly_chart(fig)

    elif analysis_type == "Combined Analysis":
        st.sidebar.subheader("Plot Type")
        plot_type = st.sidebar.selectbox("Select plot type", ["Bar Plot", "Heatmap", "Scatter Plot"])
        st.sidebar.subheader("Topic and User Selection")
        selected_topics = st.sidebar.multiselect('Select topics (optional)', list(unique_topics))
        selected_users = st.sidebar.multiselect('Select users (optional)', list(unique_usernames))
        st.sidebar.subheader("Metric Selection")
        columns_to_plot = ['engagement_rate', 'avg_engagement', 'favourites_count', 'replies_count', 'reblogs_count', 'sentiment_score']
        selected_column = st.sidebar.selectbox('Select metric to plot', columns_to_plot, index=0)
        if not selected_topics and not selected_users:
            st.warning("Please select at least one topic or one user.")
        else:
            # Filter data based on selections
            filtered_df = df_unique.copy()
            if selected_topics:
                filtered_df = filtered_df[filtered_df['topics'].apply(lambda x: any(topic in x for topic in selected_topics))]
            if selected_users:
                filtered_df = filtered_df[filtered_df['account_username'].isin(selected_users)]

            if filtered_df.empty:
                st.warning("No data available for the selected combination of topics and/or users.")
            else:
                if plot_type == "Bar Plot":
                    st.write("Graph Description: Bar plot showing average engagement metrics for selected users and/or topics.")
                    
                    if selected_users and selected_topics:
                        avg_engagement = filtered_df.groupby(['account_username', 'topics'])[selected_column].mean().reset_index()
                        fig = go.Figure()
                        for user in selected_users:
                            user_data = avg_engagement[avg_engagement['account_username'] == user]
                            fig.add_trace(go.Bar(x=user_data['topics'], y=user_data[selected_column], name=user))
                        fig.update_layout(barmode='group', xaxis_title='Topics', yaxis_title=selected_column)
                    elif selected_users:
                        avg_engagement = filtered_df.groupby('account_username')[selected_column].mean().reset_index()
                        fig = go.Figure(go.Bar(x=avg_engagement['account_username'], y=avg_engagement[selected_column]))
                        fig.update_layout(xaxis_title='Users', yaxis_title=selected_column)
                    else:
                        avg_engagement = filtered_df.groupby('topics')[selected_column].mean().reset_index()
                        fig = go.Figure(go.Bar(x=avg_engagement['topics'], y=avg_engagement[selected_column]))
                        fig.update_layout(xaxis_title='Topics', yaxis_title=selected_column)

                    fig.update_layout(
                        title=f'Average {selected_column} by {"User and Topic" if selected_users and selected_topics else "User" if selected_users else "Topic"}',
                        width=1000,
                        height=600
                    )
                    st.plotly_chart(fig)

                elif plot_type == "Heatmap":
                    st.write("Graph Description: Heatmap showing engagement metrics for selected users and/or topics.")
                    
                    if selected_users and selected_topics:
                        pivot_df = filtered_df.pivot_table(values=selected_column, index='account_username', columns='topics', aggfunc='mean')
                    elif selected_users:
                        pivot_df = filtered_df.pivot_table(values=selected_column, index='account_username', columns='topics', aggfunc='mean')
                    else:
                        pivot_df = filtered_df.pivot_table(values=selected_column, index='topics', columns='account_username', aggfunc='mean')

                    fig = go.Figure(data=go.Heatmap(
                        z=pivot_df.values,
                        x=pivot_df.columns,
                        y=pivot_df.index,
                        colorscale='Viridis'
                    ))

                    fig.update_layout(
                        title=f'Heatmap of {selected_column} by {"User and Topic" if selected_users and selected_topics else "User" if selected_users else "Topic"}',
                        xaxis_title='Topics' if selected_users else 'Users',
                        yaxis_title='Users' if selected_users else 'Topics',
                        width=1000,
                        height=600
                    )
                    st.plotly_chart(fig)

                elif plot_type == "Scatter Plot":
                    st.write("Graph Description: Scatter plot showing engagement metrics over time for selected users and/or topics.")
                    
                    fig = go.Figure()
                    if selected_users and selected_topics:
                        for user in selected_users:
                            user_data = filtered_df[filtered_df['account_username'] == user]
                            for topic in selected_topics:
                                topic_data = user_data[user_data['topics'].apply(lambda x: topic in x)]
                                if not topic_data.empty:
                                    fig.add_trace(go.Scatter(x=topic_data['created_at_iso'], y=topic_data[selected_column], mode='markers', name=f'{user} - {topic}'))
                    elif selected_users:
                        for user in selected_users:
                            user_data = filtered_df[filtered_df['account_username'] == user]
                            fig.add_trace(go.Scatter(x=user_data['created_at_iso'], y=user_data[selected_column], mode='markers', name=user))
                    else:
                        for topic in selected_topics:
                            topic_data = filtered_df[filtered_df['topics'].apply(lambda x: topic in x)]
                            fig.add_trace(go.Scatter(x=topic_data['created_at_iso'], y=topic_data[selected_column], mode='markers', name=topic))

                    fig.update_layout(
                        title=f'{selected_column} Over Time by {"User and Topic" if selected_users and selected_topics else "User" if selected_users else "Topic"}',
                        xaxis_title='Date',
                        yaxis_title=selected_column,
                        width=1000,
                        height=600
                    )
                    st.plotly_chart(fig)

                # Display summary statistics
                st.subheader("Summary Statistics")
                if selected_users and selected_topics:
                    summary_stats = filtered_df.groupby(['account_username', 'topics'])[selected_column].agg(['mean', 'median', 'min', 'max']).reset_index()
                elif selected_users:
                    summary_stats = filtered_df.groupby('account_username')[selected_column].agg(['mean', 'median', 'min', 'max']).reset_index()
                else:
                    summary_stats = filtered_df.groupby('topics')[selected_column].agg(['mean', 'median', 'min', 'max']).reset_index()
                st.write(summary_stats)

elif main_option=='New page':
    st.subheader("New page")
    