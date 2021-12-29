import gpflow
import streamlit as st
import numpy as np

from back_connection import drawNextSample, getAcqfunctions, sendSample, startExperiment, loadExperiment, drawRandomSample
from htmlComponent import disableWidget

from frontutils import filterdict
from frontGP import frontGP

import os

#Temporal
k_hps = ["Lengthscales", "Variance", "Noise Variance"]

st.set_page_config(page_title="MOO Easy Tool", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

if 'acqfunctions' not in st.session_state:
    getAcqfunctions(st.session_state)

### Initialization of the state
if 'selected' not in st.session_state:
	st.session_state.selected = False

if 'input_error' not in st.session_state:
	st.session_state.input_error = False

if 'X' not in st.session_state:
    st.session_state.X = []
if 'Y' not in st.session_state:
    st.session_state.Y = []

if 'n_ins' not in st.session_state:
    st.session_state.n_ins=1

st.title("MOO Easy Tool")

st.write("Want to automate your multi objective optimization procedure? Check [our API]("+os.getenv('backendURL', "http://0.0.0.0:8003/")+"docs)")
st.write("Check some examples on [our GitHub](https://github.com/MiguelTaibo/DashboardMOO/blob/master/examples)")

### Name section
c1,bt_load_exp,bt_start_test,c4 = st.columns([2,1,1,2])

if not st.session_state.selected:
    if bt_load_exp.button("Load exp", help="If you already started an experiment you can recap it with its name"):
        loadExperiment(st.session_state)
        if 'input_names' not in st.session_state or len(st.session_state.input_names)!=st.session_state.n_ins:
            st.session_state.input_names = [None for _ in range(st.session_state.n_ins)]
    if bt_start_test.button("Start exp", help="Start a new experiment with hyperparameters below"):
        startExperiment(st.session_state)
    if st.session_state.input_error:
        st.error(st.session_state.input_error_text)

if not st.session_state.selected:
    st.session_state.name = c1.text_input("Experiment name", max_chars=20, placeholder="Set your experiment with a name so you can recap it")
else:
    c1.markdown(disableWidget("Experiment name",st.session_state.name), unsafe_allow_html=True)

# st.write(st.session_state)


### Inputs section
with st.container():
    st.header("Algorithm Hyperparameters")

    c1,c2,c3,c4,c5 = st.columns(5)

    if not st.session_state.selected:
        st.session_state.n_ins = int(c1.number_input("N inputs", min_value=1,  format="%d", help="number of input variables to fine-tune"))
        expander = c1.expander("Input variables")

        if 'input_names' not in st.session_state or len(st.session_state.input_names)!=st.session_state.n_ins:
            st.session_state.input_names = [None for _ in range(st.session_state.n_ins)]
        if 'input_mms' not in st.session_state  or len(st.session_state.input_mms)!=st.session_state.n_ins:
            st.session_state.input_mms = [[None,None] for _ in range(st.session_state.n_ins)]

        for i in range(st.session_state.n_ins):
            st.session_state.input_names[i] = expander.text_input("Name input " + str(i+1), value="input var" + str(i+1))
            st.session_state.input_mms[i][0] = st.session_state.number_input = expander.number_input("Lower Bound " + str(i+1), value=0., format="%f")
            st.session_state.input_mms[i][1] = expander.number_input("Upper Bound " + str(i+1), value=1., format="%f")
            

        st.session_state.n_objs = c2.number_input("N objectives", min_value=1, step=1,  format="%d", help="number of objectives to maximize")
        expander = c2.expander("Output variables")

        if 'objective_names' not in st.session_state or len(st.session_state.objective_names)!=st.session_state.n_objs:
            st.session_state.objective_names = [None for _ in range(st.session_state.n_objs)]
        if 'objective_mms' not in st.session_state  or len(st.session_state.objective_mms)!=st.session_state.n_objs:
            st.session_state.objective_mms = [None for _ in range(st.session_state.n_objs)]

        for i in range(st.session_state.n_objs):
            st.session_state.objective_names[i] = expander.text_input("Name objective " + str(i+1), value="objective "+str(i+1))
            st.session_state.objective_mms[i] = expander.select_slider("Objective "+ str(i+1),options=['Minimize','Maximize'])

        st.session_state.n_cons = c3.number_input("N constrains", min_value= 0, step = 1, format="%d", help="number of constrains")


        st.session_state.acqfunct = c5.selectbox("Acquisition function", 
                    [el['name'] for el in st.session_state.acqfunctions],
                    help="Acquisition function to use in the algorithm not implemented yet")
        expander = c5.expander("Acquisitiion function hyperparameters")
        if 'acqfunct_hps' not in st.session_state:
            st.session_state.acqfunct_hps = dict()
        if not st.session_state.acqfunctions==[]:
            for hp in filterdict(st.session_state.acqfunctions,"name",st.session_state.acqfunct)[0]['hps']:
                st.session_state.acqfunct_hps[hp['name']] = expander.number_input(hp['name'], value=hp['default'], help=hp['help'], format=hp['type'], min_value=0)
    else:
        c1.markdown(disableWidget("N inputs",st.session_state.n_ins), unsafe_allow_html=True)
        expander = c1.expander("Input variables")
        for i in range(st.session_state.n_ins):
            value = st.session_state.input_names[i]+"   ["+str(st.session_state.input_mms[i][0])+" ,"+str(st.session_state.input_mms[i][1])+"]"
            expander.markdown(disableWidget("Name input " + str(i+1),value) , unsafe_allow_html=True)

        c2.markdown(disableWidget("N objectives",st.session_state.n_objs), unsafe_allow_html=True)
        expander = c2.expander("Output variables")
        for i in range(st.session_state.n_objs):
            value = st.session_state.objective_names[i]+"   ("+ ("Maximize" if st.session_state.objective_mms[i]else "Minimize")+")"
            expander.markdown(disableWidget("Name objective " + str(i+1),value), unsafe_allow_html=True)

        c3.markdown(disableWidget("N constrains", st.session_state.n_cons),unsafe_allow_html=True)

        c5.markdown(disableWidget("Acquisition function", st.session_state.acqfunct),unsafe_allow_html=True)
        expander = c5.expander("Acquisitiion function hyperparameters")
        if 'acqfunct_hps' not in st.session_state:
            st.session_state.acqfunct_hps = dict()

        for k,v in st.session_state.acqfunct_hps.items():
            expander.markdown(disableWidget(k, v),unsafe_allow_html=True)

    st.session_state.kernel = c4.selectbox("Kernel", 
                    np.array(["RBF"]),
                    help="kernel defined to use in the GP model \n not implemented yet")
    expander = c4.expander("Kernel hyperparameters")
    for hp in k_hps:
        expander.number_input(hp, min_value=0.01, step=0.01)

### Output section
with st.container():

    st.header("Algorithm Outputs")

    c1,bt_draw_sample,bt_draw_random_sample = st.columns([1,1,1])
    c1.subheader("Evaluate function with the following values")

    if st.session_state.selected:
        if bt_draw_sample.button("Draw sample"):
            drawNextSample(st.session_state)
        if bt_draw_random_sample.button("Draw random sample"):
            drawRandomSample(st.session_state)


    if 'next_x' not in st.session_state or st.session_state.next_x is None or len(st.session_state.next_x)!=st.session_state.n_ins:
        st.session_state.next_x = [0. for _ in range(st.session_state.n_ins)]
    cols = st.columns(st.session_state.n_ins)
    for idx, col in enumerate(cols):
    ## This is for being unable to change 
        # col.markdown(disableWidget(st.session_state.input_names[idx],st.session_state.next_x[idx]), unsafe_allow_html=True)
        st.session_state.next_x[idx] = col.number_input(st.session_state.input_names[idx],value=st.session_state.next_x[idx], format="%.6f")


    c1,bt_send_sample = st.columns([1,2])
    c1.subheader("Set the results of evaluating the function here")

    if st.session_state.selected:
        if bt_send_sample.button("Send sample"):
            sendSample(st.session_state)

    if 'next_y' not in st.session_state or len(st.session_state.next_y)!=st.session_state.n_objs:
        st.session_state.next_y = [None for _ in range(st.session_state.n_objs)]
    
    cols = st.columns(st.session_state.n_objs)
    for idx, col in enumerate(cols):
        st.session_state.next_y[idx] = col.number_input(st.session_state.objective_names[idx], format="%f")

### Graph section
with st.container():
    st.header("Graphs")

    if st.session_state.X is None or st.session_state.X == []:
        st.subheader("Get at least one sample to visualize graphs")
    else:
        kernel = gpflow.kernels.SquaredExponential()
        GP = frontGP(
            st.session_state.n_objs,
            st.session_state.n_cons,
            st.session_state.n_ins,
            np.array(st.session_state.input_mms)[:,0],
            np.array(st.session_state.input_mms)[:,1],
            kernel = kernel,
            X = np.array(st.session_state.X),
            Y = np.array(st.session_state.Y))
        GP.updateGPR()
        GP.optimizeKernel()
        

        c1,_,c2,_ = st.columns([1,1,1,4])
        graph_select = c1.select_slider('', options=["GP model", "Pareto sets"], key="graph selector")
        c2.download_button(
            label="Download Paretos CSV",
            data=GP.dlParetos(st.session_state),
            file_name=st.session_state.name+'_paretos.csv',
            mime='text/csv',
        )

        if graph_select=="GP model":
            fig, axs = GP.plot()
            if GP.O>1 and GP.d>1:
                for i in range(GP.O):
                    axs[i,0].set_ylabel(st.session_state.objective_names[i])
                for i in range(GP.d):
                    axs[GP.O-1,i].set_xlabel(st.session_state.input_names[i])

            c1,c2,c3 = st.columns([1,2*GP.d,1])

            c2.pyplot(fig)
        elif graph_select=="Pareto sets":

            c1,c2,c3,c4,c5 = st.columns([2,3,2,3,2])
            
            c2.header("Pareto Set")
            c4.header("Pareto Front")
            c4.selectbox("X axis", options = st.session_state.objective_names, index=0, key="frontx")
            c4.selectbox("Y axis", options = st.session_state.objective_names, index=1, key = "fronty")
            c2.selectbox("X axis", options = st.session_state.input_names, index=0, key="setx")
            c2.selectbox("Y axis", options = st.session_state.input_names, index=1, key="sety")
            

            fig1, fig2 = GP.plotParetos(st.session_state)
            c1,c2,c3,c4 = st.columns([2,4,4,2])
            c2.pyplot(fig1)
            c3.pyplot(fig2)
