import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import tensorflow as tf

model_svc = pickle.load(open('model_svc.pkl', 'rb'))
model_tree = pickle.load(open('model_svc.pkl', 'rb'))
model_net = tf.keras.models.load_model("model_net.h5")

# ----------- General things
st.set_page_config(layout="wide")
st.header("Customer Hotel Checkin Predictor")
st.write("Please fill in all the customer details to predict whether they would check in or not.")

# ----------- Sidebar

st.sidebar.header('The Marker Hotel')

image = Image.open('hotel.png')

st.sidebar.image(image, use_column_width = 'always')

values = ['Neural Net', 'LinearSVC', 'Random Forest']
default_ix = values.index('Neural Net')
page = st.sidebar.selectbox('Prediction Method', values, index=default_ix)

st.sidebar.markdown("""---""")
st.sidebar.write("Created by [Kanishk Kumar](https://www.linkedin.com/in/kanishk-kumar11/)")
st.sidebar.write("Source code can be found [here.](https://github.com/Kanishk-Kumar/hotel_cust_cls)")
st.sidebar.write("A Jupyter Notebook explaining the model building can be found [here.](https://github.com/Kanishk-Kumar/hotel_cust_cls/blob/main/bookings_notebook.ipynb)")


if page == "Neural Net":
    
    with st.form(key='data', clear_on_submit=True):
        
        # ----------- Inputs
    
        columns = st.columns([1, 1, 1, 1])
        
        Age = columns[0].number_input('Age of the customer:', min_value = 0, step=1, help = 'Should be more than one.')
        DaysSinceCreation = columns[0].number_input('Number of days since the customer record was created:', min_value = 0, step=1)
        AverageLeadTime = columns[0].number_input('The average number of days elapsed between the customer\'s booking date and arrival date:', min_value = 0, step=1)
        LodgingRevenue = columns[0].number_input('Total amount spent on lodging expenses by the customer (in Euros):', min_value = 0, step=1)
        OtherRevenue = columns[0].number_input('Total amount spent on other expenses by the customer (in Euros)', min_value = 0, step=1)
        BookingsCanceled = columns[0].number_input('Number of bookings the customer made but subsequently canceled:', min_value = 0, step=1)
        BookingsNoShowed = columns[1].number_input('Number of bookings the customer made but never showed up:', min_value = 0, step=1)
        PersonsNights = columns[1].number_input('The total number of persons/nights that the costumer stayed at the hotel:', min_value = 0, step=1, help = 'This value is calculated by summing all customers checked-in bookings’ persons/nights. Person/nights of each booking is the result of the multiplication of the number of staying nights by the sum of adults and children.')
        RoomNights = columns[1].number_input('Total of room/nights the customer stayed at the hotel:', min_value = 0, step=1, help = 'Room/nights are the multiplication of the number of rooms of each booking by the number of nights of the booking.')
        DaysSinceLastStay = columns[1].number_input('The number of days elapsed since the customer\'s last arrival date:', min_value = 0, step=1)
        DaysSinceFirstStay = columns[1].number_input('The number of days elapsed since the customer\'s first arrival date:', min_value = 0, step=1)
        SRHighFloor = columns[1].radio('Does the customer usually asks for a room on a higher floor?', ['Yes', 'No'])
        SRLowFloor = columns[2].radio('Does the customer usually asks for a lower floor?', ['Yes', 'No'])
        SRAccessibleRoom = columns[2].radio('Does the customer usually asks for an an accessible room?', ['Yes', 'No'])
        SRMediumFloor = columns[2].radio('Does the customer usually asks for a room on a middle floor?', ['Yes', 'No'])
        SRBathtub = columns[2].radio('Does the customer usually asks for a room with a bathtub?', ['Yes', 'No'])
        SRShower = columns[2].radio('Does the customer usually asks for a room with a shower?', ['Yes', 'No'])
        SRCrib = columns[2].radio('Does the customer usually asks for a crib?', ['Yes', 'No'])
        SRKingSizeBed = columns[3].radio('Does the customer usually asks for a room with a king-size bed?', ['Yes', 'No'])
        SRTwinBed = columns[3].radio('Does the customer usually asks for a room with a twin bed?', ['Yes', 'No'])
        SRNearElevator = columns[3].radio('Does the customer usually asks for a room near the elevator?', ['Yes', 'No'])
        SRAwayFromElevator = columns[3].radio('Does the customer usually asks for a room away from the elevator?', ['Yes', 'No'])
        SRNoAlcoholInMiniBar = columns[3].radio('Does the customer usually asks for a room with no alcohol in the mini-bar?', ['Yes', 'No'])
        SRQuietRoom = columns[3].radio('Does the customer usually asks for a room away from the noise?', ['Yes', 'No'])
        
        data = {'Age' : [Age],
                'DaysSinceCreation' : [DaysSinceCreation],
                'AverageLeadTime' : [AverageLeadTime],
                'LodgingRevenue' : [LodgingRevenue],
                'OtherRevenue' : [OtherRevenue],
                'BookingsCanceled' : [BookingsCanceled],
                'BookingsNoShowed' : [BookingsNoShowed],
                'PersonsNights' : [PersonsNights],
                'RoomNights' : [RoomNights],
                'DaysSinceLastStay' : [DaysSinceLastStay],
                'DaysSinceFirstStay' : [DaysSinceFirstStay],
                'SRHighFloor' : [SRHighFloor],
                'SRLowFloor' : [SRLowFloor],
                'SRAccessibleRoom' : [SRAccessibleRoom],
                'SRMediumFloor' : [SRMediumFloor],
                'SRBathtub' : [SRBathtub],
                'SRShower' : [SRShower],
                'SRCrib' : [SRCrib],
                'SRKingSizeBed' : [SRKingSizeBed],
                'SRTwinBed' : [SRTwinBed],
                'SRNearElevator' : [SRNearElevator],
                'SRAwayFromElevator' : [SRAwayFromElevator],
                'SRNoAlcoholInMiniBar' : [SRNoAlcoholInMiniBar],
                'SRQuietRoom' : [SRQuietRoom]}
        
        data = pd.DataFrame(data)
        
        data.replace({'SRHighFloor' : {'Yes' : 1, 'No' : 0},
                      'SRLowFloor' : {'Yes' : 1, 'No' : 0},
                      'SRAccessibleRoom' : {'Yes' : 1, 'No' : 0},
                      'SRMediumFloor' : {'Yes' : 1, 'No' : 0},
                      'SRBathtub' : {'Yes' : 1, 'No' : 0},
                      'SRShower' : {'Yes' : 1, 'No' : 0},
                      'SRCrib' : {'Yes' : 1, 'No' : 0},
                      'SRKingSizeBed' : {'Yes' : 1, 'No' : 0},
                      'SRTwinBed' : {'Yes' : 1, 'No' : 0},
                      'SRNearElevator' : {'Yes' : 1, 'No' : 0},
                      'SRAwayFromElevator' : {'Yes' : 1, 'No' : 0},
                      'SRNoAlcoholInMiniBar' : {'Yes' : 1, 'No' : 0},
                      'SRQuietRoom' : {'Yes' : 1, 'No' : 0}}, inplace=True)
        
        data = data.astype(float)
        
        submit_button = st.form_submit_button(label='Predict')
        
        if submit_button:
            warn1 = 0
            warn2 = 0
            for x in data.iloc[:, 0:11].columns:
                if data.loc[0, x] == 0:
                    warn1 = 1
                if data.loc[0, 'Age'] == 0:
                    warn2 = 1
            if warn2 == 1:
                st.error('Age cannot be zero.')
            else:
                if warn1 == 1:
                    st.error('Some of the data you entered are zero. To get more than 99% accuracy, make sure all the entered data is correct.')
                else:
                    st.info('To get more than 99% accuracy, make sure all the entered data is correct.')
                st.subheader('Prediction:') 
                prediction = model_net.predict(data)
                prediction = tf.greater(prediction, .5)
                if prediction.numpy().astype(int)[0] == 0:
                    st.success("**This customer will check in.**")
                else:
                    st.success("**This customer will not check in.**")

elif page == "LinearSVC":
    
    with st.form(key='data', clear_on_submit=True):
        
        # ----------- Inputs
    
        columns = st.columns([1, 1, 1, 1])
        
        Age = columns[0].number_input('Age of the customer:', min_value = 0, step=1, help = 'Should be more than one.')
        DaysSinceCreation = columns[0].number_input('Number of days since the customer record was created:', min_value = 0, step=1)
        AverageLeadTime = columns[0].number_input('The average number of days elapsed between the customer\'s booking date and arrival date:', min_value = 0, step=1)
        LodgingRevenue = columns[0].number_input('Total amount spent on lodging expenses by the customer (in Euros):', min_value = 0, step=1)
        OtherRevenue = columns[0].number_input('Total amount spent on other expenses by the customer (in Euros)', min_value = 0, step=1)
        BookingsCanceled = columns[0].number_input('Number of bookings the customer made but subsequently canceled:', min_value = 0, step=1)
        BookingsNoShowed = columns[1].number_input('Number of bookings the customer made but never showed up:', min_value = 0, step=1)
        PersonsNights = columns[1].number_input('The total number of persons/nights that the costumer stayed at the hotel:', min_value = 0, step=1, help = 'This value is calculated by summing all customers checked-in bookings’ persons/nights. Person/nights of each booking is the result of the multiplication of the number of staying nights by the sum of adults and children.')
        RoomNights = columns[1].number_input('Total of room/nights the customer stayed at the hotel:', min_value = 0, step=1, help = 'Room/nights are the multiplication of the number of rooms of each booking by the number of nights of the booking.')
        DaysSinceLastStay = columns[1].number_input('The number of days elapsed since the customer\'s last arrival date:', min_value = 0, step=1)
        DaysSinceFirstStay = columns[1].number_input('The number of days elapsed since the customer\'s first arrival date:', min_value = 0, step=1)
        SRHighFloor = columns[1].radio('Does the customer usually asks for a room on a higher floor?', ['Yes', 'No'])
        SRLowFloor = columns[2].radio('Does the customer usually asks for a lower floor?', ['Yes', 'No'])
        SRAccessibleRoom = columns[2].radio('Does the customer usually asks for an an accessible room?', ['Yes', 'No'])
        SRMediumFloor = columns[2].radio('Does the customer usually asks for a room on a middle floor?', ['Yes', 'No'])
        SRBathtub = columns[2].radio('Does the customer usually asks for a room with a bathtub?', ['Yes', 'No'])
        SRShower = columns[2].radio('Does the customer usually asks for a room with a shower?', ['Yes', 'No'])
        SRCrib = columns[2].radio('Does the customer usually asks for a crib?', ['Yes', 'No'])
        SRKingSizeBed = columns[3].radio('Does the customer usually asks for a room with a king-size bed?', ['Yes', 'No'])
        SRTwinBed = columns[3].radio('Does the customer usually asks for a room with a twin bed?', ['Yes', 'No'])
        SRNearElevator = columns[3].radio('Does the customer usually asks for a room near the elevator?', ['Yes', 'No'])
        SRAwayFromElevator = columns[3].radio('Does the customer usually asks for a room away from the elevator?', ['Yes', 'No'])
        SRNoAlcoholInMiniBar = columns[3].radio('Does the customer usually asks for a room with no alcohol in the mini-bar?', ['Yes', 'No'])
        SRQuietRoom = columns[3].radio('Does the customer usually asks for a room away from the noise?', ['Yes', 'No'])
        
        data = {'Age' : [Age],
                'DaysSinceCreation' : [DaysSinceCreation],
                'AverageLeadTime' : [AverageLeadTime],
                'LodgingRevenue' : [LodgingRevenue],
                'OtherRevenue' : [OtherRevenue],
                'BookingsCanceled' : [BookingsCanceled],
                'BookingsNoShowed' : [BookingsNoShowed],
                'PersonsNights' : [PersonsNights],
                'RoomNights' : [RoomNights],
                'DaysSinceLastStay' : [DaysSinceLastStay],
                'DaysSinceFirstStay' : [DaysSinceFirstStay],
                'SRHighFloor' : [SRHighFloor],
                'SRLowFloor' : [SRLowFloor],
                'SRAccessibleRoom' : [SRAccessibleRoom],
                'SRMediumFloor' : [SRMediumFloor],
                'SRBathtub' : [SRBathtub],
                'SRShower' : [SRShower],
                'SRCrib' : [SRCrib],
                'SRKingSizeBed' : [SRKingSizeBed],
                'SRTwinBed' : [SRTwinBed],
                'SRNearElevator' : [SRNearElevator],
                'SRAwayFromElevator' : [SRAwayFromElevator],
                'SRNoAlcoholInMiniBar' : [SRNoAlcoholInMiniBar],
                'SRQuietRoom' : [SRQuietRoom]}
        
        data = pd.DataFrame(data)
        
        data.replace({'SRHighFloor' : {'Yes' : 1, 'No' : 0},
                      'SRLowFloor' : {'Yes' : 1, 'No' : 0},
                      'SRAccessibleRoom' : {'Yes' : 1, 'No' : 0},
                      'SRMediumFloor' : {'Yes' : 1, 'No' : 0},
                      'SRBathtub' : {'Yes' : 1, 'No' : 0},
                      'SRShower' : {'Yes' : 1, 'No' : 0},
                      'SRCrib' : {'Yes' : 1, 'No' : 0},
                      'SRKingSizeBed' : {'Yes' : 1, 'No' : 0},
                      'SRTwinBed' : {'Yes' : 1, 'No' : 0},
                      'SRNearElevator' : {'Yes' : 1, 'No' : 0},
                      'SRAwayFromElevator' : {'Yes' : 1, 'No' : 0},
                      'SRNoAlcoholInMiniBar' : {'Yes' : 1, 'No' : 0},
                      'SRQuietRoom' : {'Yes' : 1, 'No' : 0}}, inplace=True)
        
        data = data.astype(float)
        
        submit_button = st.form_submit_button(label='Predict')
        
        if submit_button:
            warn1 = 0
            warn2 = 0
            for x in data.iloc[:, 0:11].columns:
                if data.loc[0, x] == 0:
                    warn1 = 1
                if data.loc[0, 'Age'] == 0:
                    warn2 = 1
            if warn2 == 1:
                st.error('Age cannot be zero.')
            else:
                if warn1 == 1:
                    st.error('Some of the data you entered are zero. To get more than 99% accuracy, make sure all the entered data is correct.')
                else:
                    st.info('To get more than 99% accuracy, make sure all the entered data is correct.')
                st.subheader('Prediction:') 
                prediction = model_svc.predict(data)
                if prediction[0] == 0:
                    st.success("**This customer will check in.**")
                else:
                    st.success("**This customer will not check in.**")

elif page == "Random Forest":

    with st.form(key='data', clear_on_submit=True):
        
        # ----------- Inputs
    
        columns = st.columns([1, 1, 1, 1])
        
        Age = columns[0].number_input('Age of the customer:', min_value = 0, step=1, help = 'Should be more than one.')
        DaysSinceCreation = columns[0].number_input('Number of days since the customer record was created:', min_value = 0, step=1)
        AverageLeadTime = columns[0].number_input('The average number of days elapsed between the customer\'s booking date and arrival date:', min_value = 0, step=1)
        LodgingRevenue = columns[0].number_input('Total amount spent on lodging expenses by the customer (in Euros):', min_value = 0, step=1)
        OtherRevenue = columns[0].number_input('Total amount spent on other expenses by the customer (in Euros)', min_value = 0, step=1)
        BookingsCanceled = columns[0].number_input('Number of bookings the customer made but subsequently canceled:', min_value = 0, step=1)
        BookingsNoShowed = columns[1].number_input('Number of bookings the customer made but never showed up:', min_value = 0, step=1)
        PersonsNights = columns[1].number_input('The total number of persons/nights that the costumer stayed at the hotel:', min_value = 0, step=1, help = 'This value is calculated by summing all customers checked-in bookings’ persons/nights. Person/nights of each booking is the result of the multiplication of the number of staying nights by the sum of adults and children.')
        RoomNights = columns[1].number_input('Total of room/nights the customer stayed at the hotel:', min_value = 0, step=1, help = 'Room/nights are the multiplication of the number of rooms of each booking by the number of nights of the booking.')
        DaysSinceLastStay = columns[1].number_input('The number of days elapsed since the customer\'s last arrival date:', min_value = 0, step=1)
        DaysSinceFirstStay = columns[1].number_input('The number of days elapsed since the customer\'s first arrival date:', min_value = 0, step=1)
        SRHighFloor = columns[1].radio('Does the customer usually asks for a room on a higher floor?', ['Yes', 'No'])
        SRLowFloor = columns[2].radio('Does the customer usually asks for a lower floor?', ['Yes', 'No'])
        SRAccessibleRoom = columns[2].radio('Does the customer usually asks for an an accessible room?', ['Yes', 'No'])
        SRMediumFloor = columns[2].radio('Does the customer usually asks for a room on a middle floor?', ['Yes', 'No'])
        SRBathtub = columns[2].radio('Does the customer usually asks for a room with a bathtub?', ['Yes', 'No'])
        SRShower = columns[2].radio('Does the customer usually asks for a room with a shower?', ['Yes', 'No'])
        SRCrib = columns[2].radio('Does the customer usually asks for a crib?', ['Yes', 'No'])
        SRKingSizeBed = columns[3].radio('Does the customer usually asks for a room with a king-size bed?', ['Yes', 'No'])
        SRTwinBed = columns[3].radio('Does the customer usually asks for a room with a twin bed?', ['Yes', 'No'])
        SRNearElevator = columns[3].radio('Does the customer usually asks for a room near the elevator?', ['Yes', 'No'])
        SRAwayFromElevator = columns[3].radio('Does the customer usually asks for a room away from the elevator?', ['Yes', 'No'])
        SRNoAlcoholInMiniBar = columns[3].radio('Does the customer usually asks for a room with no alcohol in the mini-bar?', ['Yes', 'No'])
        SRQuietRoom = columns[3].radio('Does the customer usually asks for a room away from the noise?', ['Yes', 'No'])
        
        data = {'Age' : [Age],
                'DaysSinceCreation' : [DaysSinceCreation],
                'AverageLeadTime' : [AverageLeadTime],
                'LodgingRevenue' : [LodgingRevenue],
                'OtherRevenue' : [OtherRevenue],
                'BookingsCanceled' : [BookingsCanceled],
                'BookingsNoShowed' : [BookingsNoShowed],
                'PersonsNights' : [PersonsNights],
                'RoomNights' : [RoomNights],
                'DaysSinceLastStay' : [DaysSinceLastStay],
                'DaysSinceFirstStay' : [DaysSinceFirstStay],
                'SRHighFloor' : [SRHighFloor],
                'SRLowFloor' : [SRLowFloor],
                'SRAccessibleRoom' : [SRAccessibleRoom],
                'SRMediumFloor' : [SRMediumFloor],
                'SRBathtub' : [SRBathtub],
                'SRShower' : [SRShower],
                'SRCrib' : [SRCrib],
                'SRKingSizeBed' : [SRKingSizeBed],
                'SRTwinBed' : [SRTwinBed],
                'SRNearElevator' : [SRNearElevator],
                'SRAwayFromElevator' : [SRAwayFromElevator],
                'SRNoAlcoholInMiniBar' : [SRNoAlcoholInMiniBar],
                'SRQuietRoom' : [SRQuietRoom]}
        
        data = pd.DataFrame(data)
        
        data.replace({'SRHighFloor' : {'Yes' : 1, 'No' : 0},
                      'SRLowFloor' : {'Yes' : 1, 'No' : 0},
                      'SRAccessibleRoom' : {'Yes' : 1, 'No' : 0},
                      'SRMediumFloor' : {'Yes' : 1, 'No' : 0},
                      'SRBathtub' : {'Yes' : 1, 'No' : 0},
                      'SRShower' : {'Yes' : 1, 'No' : 0},
                      'SRCrib' : {'Yes' : 1, 'No' : 0},
                      'SRKingSizeBed' : {'Yes' : 1, 'No' : 0},
                      'SRTwinBed' : {'Yes' : 1, 'No' : 0},
                      'SRNearElevator' : {'Yes' : 1, 'No' : 0},
                      'SRAwayFromElevator' : {'Yes' : 1, 'No' : 0},
                      'SRNoAlcoholInMiniBar' : {'Yes' : 1, 'No' : 0},
                      'SRQuietRoom' : {'Yes' : 1, 'No' : 0}}, inplace=True)
        
        data = data.astype(float)
        
        submit_button = st.form_submit_button(label='Predict')
        
        if submit_button:
            warn1 = 0
            warn2 = 0
            for x in data.iloc[:, 0:11].columns:
                if data.loc[0, x] == 0:
                    warn1 = 1
                if data.loc[0, 'Age'] == 0:
                    warn2 = 1
            if warn2 == 1:
                st.error('Age cannot be zero.')
            else:
                if warn1 == 1:
                    st.error('Some of the data you entered are zero. To get more than 99% accuracy, make sure all the entered data is correct.')
                else:
                    st.info('To get more than 99% accuracy, make sure all the entered data is correct.')
                st.subheader('Prediction:') 
                prediction = model_tree.predict(data)
                if prediction[0] == 0:
                    st.success("**This customer will check in.**")
                else:
                    st.success("**This customer will not check in.**")
