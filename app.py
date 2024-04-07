import streamlit as st
import joblib


# Load model
model = joblib.load('models/svm_project1.joblib')


# Tiêu đề của ứng dụng
st.title('Review Restaurant')

# Hiển thị trình nhập dữ liệu để người dùng nhập số
number = st.number_input(label='Nhập ID nhà hàng')

# nhập dữ liệu comment
comment = st.text_area(label='Nhập comment của bạn')

# Kiểm tra xem người dùng đã nhập số hay chưa
if st.button('Submit'):
    st.write(f'id: {number}, comment: {comment}')
    st.write('Đánh giá của bạn là: ' + str(model.predict([comment])[0]))
