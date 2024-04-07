import streamlit as st
import joblib
import preprocessing_text as pt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt


from wordcloud import WordCloud

# Load model
model = joblib.load('models/svm_project1.joblib')

# load data
df = pd.read_csv('Data/2_Reviews.csv')

# Tiêu đề của ứng dụng
st.title('Review Restaurant')

# Hiển thị trình nhập dữ liệu để người dùng nhập số
id = st.number_input(label='Nhập ID nhà hàng')

# nhập dữ liệu comment
comment = st.text_area(label='Nhập comment của bạn')

# Kiểm tra xem người dùng đã nhập số hay chưa
if st.button('Submit'):
    # st.write(f'id: {number}, comment: {comment}')
    data = df[df['IDRestaurant'] == int(id)]
    st.write('Đánh giá của bạn là: ' + str(model.predict([comment])[0]))
    # predict review
    pred = model.predict(data['Comment'])
    res = np.unique(pred, return_counts=True)
    # add column pred
    data['pred'] = pred

    # histogram char
    x_values = data['Rating'].value_counts().index
    y_values = data['Rating'].value_counts().values
    fig_hist = go.Figure(go.Bar(x = x_values, y = y_values))
    fig_hist.update_layout(title={'text':'Tổng quan các lượt ratings và phản hồi về nhà hàng', 'font': {'size': 30}}, xaxis={'title': 'Rating'}, yaxis={'title': 'Số lượt đánh giá'})

    # pie chart
    fig_pie = [go.Pie(labels=res[0], values=res[1])]
    fig_pie = go.Figure(fig_pie)
    fig_pie.update_layout(title={'text':'Tỉ lệ các loại đánh giá', 'font': {'size': 30}})

    # wordcloud negative
    text_neg = ' '.join(data[data['pred'] == 'Negative']['Comment'])
    text_neg = pt.optimized_process_text(text_neg, pt.stopwords_lst)
    wc_neg = WordCloud(width=800, height=400, background_color='white').generate(text_neg)
    fig_wc_neg = plt.figure(figsize=(10, 5))
    plt.imshow(wc_neg, interpolation='bilinear')
    plt.title('Wordcloud negative', fontsize=20, fontweight='bold')
    plt.axis('off')
    
    # wordcloud positive
    text_pos = ' '.join(data[data['pred'] == 'Positive']['Comment'])
    text_pos = pt.optimized_process_text(text_pos, pt.stopwords_lst)
    wc_pos = WordCloud(width=800, height=400, background_color='white').generate(text_pos)
    fig_wc_pos = plt.figure(figsize=(10, 5))
    plt.imshow(wc_neg, interpolation='bilinear')
    plt.title('Wordcloud positive', fontsize=20, fontweight='bold')
    plt.axis('off')
    
    # display
    st.plotly_chart(fig_hist, use_container_width=True)
    st.plotly_chart(fig_pie, use_container_width=True)
    st.pyplot(fig_wc_neg, use_container_width=True)
    st.pyplot(fig_wc_pos, use_container_width=True)


