import streamlit as st
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# --- Streamlit App Title ---
st.title("⚡ Energy Meter - Machine Learning Analysis")

# --- Upload Dataset ---
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload your Energy Meter CSV", type=["csv"])

if uploaded_file is not None:
    names = ['Voltage', 'Current', 'Power', 'class']
    dataset = read_csv(uploaded_file, names=names)

    st.subheader("📊 Dataset Preview")
    st.write(dataset.head(10))

    st.subheader("Dataset Shape")
    st.write(dataset.shape)

    st.subheader("Class Distribution")
    st.write(dataset.groupby('class').size())

    st.subheader("Dataset Statistics")
    st.write(dataset.describe())

    # --- Visualization Section ---
    st.subheader("🔍 Data Visualization")
    if st.checkbox("Show Bar Plot"):
        dataset.plot(kind='bar', subplots=True, layout=(2, 2))
        st.pyplot(pyplot)

    if st.checkbox("Show Histogram"):
        dataset.hist()
        st.pyplot(pyplot)

    if st.checkbox("Show Scatter Matrix"):
        scatter_matrix(dataset)
        st.pyplot(pyplot)

    # --- Machine Learning Section ---
    st.subheader("🤖 ML Model Comparison")

    array = dataset.values
    X = array[:, 0:3]
    y = array[:, 3]
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, y, test_size=0.20, random_state=1, shuffle=True
    )

    models = [
        ('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
        ('LDA', LinearDiscriminantAnalysis()),
        ('KNN', KNeighborsClassifier()),
        ('CART', DecisionTreeClassifier()),
        ('NB', GaussianNB()),
        ('SVM', SVC(gamma='auto')),
    ]

    results = []
    names = []
    res = []

    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=None)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        res.append(cv_results.mean())

    st.write("### Model Accuracy Results")
    for i in range(len(names)):
        st.write(f"{names[i]}: {res[i]:.4f}")

    fig, ax = pyplot.subplots()
    ax.bar(names, res, color='maroon', width=0.6)
    ax.set_ylim(0.0, 1.0)
    ax.set_title('Algorithm Comparison')
    st.pyplot(fig)

else:
    st.info("👆 Please upload a CSV file to begin.")
