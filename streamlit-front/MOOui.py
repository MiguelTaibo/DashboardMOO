from sqlalchemy.sql.expression import select
import streamlit as st
import numpy as np
import requests

from back_connection import sendSample, startExperiment, loadExperiment, drawRandomSample
from htmlComponent import disableWidget

#Temporal
k_hps = ["Lengthscales", "Variance", "Noise Variance"]

### Initialization of the state
if 'selected' not in st.session_state:
	st.session_state.selected = False

if 'n_ins' not in st.session_state:
    st.session_state.n_ins=1

st.set_page_config(layout="wide")
st.title("MOO Easy Tool")

c1,bt_load_exp,c3 = st.columns([2,1,3])

if not st.session_state.selected:
    if bt_load_exp.button("Load exp", help="If you already started an experiment you can recap it with its name"):
        loadExperiment(st.session_state)
        if 'input_names' not in st.session_state or len(st.session_state.input_names)!=st.session_state.n_ins:
            st.session_state.input_names = [None for _ in range(st.session_state.n_ins)]

if not st.session_state.selected:
    st.session_state.name = c1.text_input("Experiment name", max_chars=20, placeholder="Set your experiment with a name so you can recap it")
else:
    c1.markdown(disableWidget("Experiment name",st.session_state.name), unsafe_allow_html=True)



with st.container():
    c1,bt_start_test = st.columns([3,1])
    c1.header("Algorithm Hyperparameters")
    if not st.session_state.selected:
        if bt_start_test.button("Start exp", help="Start a new experiment with hyperparameters below"):
            startExperiment(st.session_state)

    c1,c2,c3,c4,c5,c6 = st.columns(6)

    if not st.session_state.selected:
        st.session_state.n_ins = int(c1.number_input("N inputs", min_value=1, step=1, help="number of input variables to fine-tune"))
        expander = c1.expander("Input variables")
        st.session_state.input_names = [expander.text_input("Name input " + str(i+1), value="input var" + str(i+1)) for i in range(st.session_state.n_ins)]
    
        st.session_state.n_objs = c2.number_input("N objectives", min_value=1, step=1, help="number of objectives to maximize")
        expander = c2.expander("Output variables")
        st.session_state.objective_names = [expander.text_input("Name objective " + str(i+1), value="objective "+str(i+1)) for i in range(st.session_state.n_objs)]

        st.session_state.n_cons = c3.number_input("N constrains", min_value= 0, step = 1, help="number of constrains")

    else:
        c1.markdown(disableWidget("N inputs",st.session_state.n_ins), unsafe_allow_html=True)
        expander = c1.expander("Input variables")
        for i in range(st.session_state.n_ins):
            expander.markdown(disableWidget("Name input " + str(i+1),st.session_state.input_names[i]), unsafe_allow_html=True)

        c2.markdown(disableWidget("N objectives",st.session_state.n_objs), unsafe_allow_html=True)
        expander = c2.expander("Output variables")
        for i in range(st.session_state.n_objs):
            expander.markdown(disableWidget("Name objective " + str(i+1),st.session_state.objective_names[i]), unsafe_allow_html=True)

        c3.markdown(disableWidget("N constrains", st.session_state.n_cons),unsafe_allow_html=True)


    st.session_state.initialIter = c4.number_input("Initial iterations", min_value= 1, step = 1, help="Number of samples randomly taken at the start of the procedure")
    st.session_state.totalIter = c4.number_input("Total iterations", min_value= 1, step = 1, help="Number iterations of the main loop of the algorithm")

    st.session_state.kernel = c5.selectbox("Kernel", 
                    np.array(["RBF"]),
                    help="kernel defined to use in the GP model \n not implemented yet")
    expander = c5.expander("Kernel hyperparameters")
    for hp in k_hps:
        expander.number_input(hp, min_value=0.01, step=0.01)

    st.session_state.acq_funct = c6.selectbox("Acquisition function", 
                    np.array(["MES", "MESMO"]),
                    help="Acquisition function to use in the algorithm \n not implemented yet")
    expander = c6.expander("Acquisitiion function hyperparameters")
    for hp in k_hps:
        expander.number_input("acq"+hp, min_value=0.01, step=0.01)

with st.container():

    st.header("Algorithm Outputs")

    c1,bt_draw_sample,bt_draw_random_sample = st.columns([1,1,1])
    c1.subheader("Evaluate function with the following values")

    if st.session_state.selected:
        if bt_draw_random_sample.button("Draw random sample"):
            drawRandomSample(st.session_state)


    if 'next_x' not in st.session_state or st.session_state.next_x is None or len(st.session_state.next_x)!=st.session_state.n_ins:
        st.session_state.next_x = [None for _ in range(st.session_state.n_ins)]
    cols = st.columns(st.session_state.n_ins)
    for idx, col in enumerate(cols):
        col.markdown(disableWidget(st.session_state.input_names[idx],st.session_state.next_x[idx]), unsafe_allow_html=True)
        
    c1,bt_send_sample = st.columns([1,2])
    c1.subheader("Set the results of evaluating the function here")

    if st.session_state.selected:
        if bt_send_sample.button("Send sample"):
            sendSample(st.session_state)

    if 'next_y' not in st.session_state or len(st.session_state.next_y)!=st.session_state.n_objs:
        st.session_state.next_y = [None for _ in range(st.session_state.n_objs)]
    
    cols = st.columns(st.session_state.n_objs)
    for idx, col in enumerate(cols):
        st.session_state.next_y[idx] = col.number_input(st.session_state.objective_names[idx])

with st.container():
    st.header("Graphs")
    st.write(st.session_state)
    st.subheader("All to do, although, as this is python we can use pyplot as in the original code :)")
    st.subheader("Nevertheless, the computations are supposed to be done in the back-end, so we do not have them here")


if st.session_state.selected:
    if bt_draw_sample.button("Draw sample"):
        pass
