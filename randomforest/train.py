from random import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
from randomforest.RandomForest import RandomForest
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # or RandomForestRegressor for regression tasks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # or other relevant metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from randomforest.models import LastTrainingResultsRandomForest
from sklearn.metrics import confusion_matrix

# data = datasets.load_breast_cancer()
# X = data.data
# y = data.target

def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

def train(X,y):
    
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=1234
        )

        # Create a Random Forest model
        model = RandomForestClassifier(n_estimators=10, random_state=42)  # You can adjust parameters

        # Train the model
        model.fit(X_train, y_train)
 

        # Make predictions on the test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        # Generate confusion matrix     
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_)
        plt.xlabel('Predicted Labels')
        plxt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

        rfTrainingModel = LastTrainingResultsRandomForest(
                accuracy = f'{accuracy * 100}%',
        )
        rfTrainingModel.save()
        # Assuming you have your model already trained and X_test contains test data
        # y_scores = model.predict_proba(X_test)[:, 1]  # Probabilities of positive class
        # n_classes = y_scores.shape[1]  # Number of classes

        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()

        # for i in range(n_classes):
        #         fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_scores[:, i])
        #         roc_auc[i] = auc(fpr[i], tpr[i])

                # plt.figure()
                # plt.plot(fpr, tpr, label='ROC curve')
                # plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
                # plt.xlabel('False Positive Rate')
                # plt.ylabel('True Positive Rate')
                # plt.title('ROC Curve')
                # plt.legend(loc='best')
                # # Save the plot as an image file
                # roc_image_path = 'path_to_save_image.png'
                # plt.savefig(roc_image_path)

                # trained_model = LastTrainingResultsRandomForest(
                #         roc_curve_image = roc_image_path
                # ) 

              
        
        kfda(X,y)
        # clf = RandomForest(n_trees=20)
        # clf.fit(X_train, y_train)
        # predictions = clf.predict(X_test)

        # acc =  accuracy(y_test, y_pred)
        # print(acc)


def kfda(X, y):
        # X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0, random_state=42)

        # Initialize and fit the LinearDiscriminantAnalysis model

        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=1234
        )


        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")  

        kfdaTrainingModel = LastTrainingResultsKFDA(
                accuracy = f'{accuracy * 100}%',
                number_of_cells = X.shape[0]
        )
        kfdaTrainingModel.save()  

        # Project the data onto the first discriminant direction
        X_projected = lda.transform(X)

        # Plot the original data and the projected data
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Class 0', marker='o')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1', marker='x')
        plt.title('Original Data')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(X_projected[y == 0], np.zeros_like(X_projected[y == 0]), label='Class 0', marker='o')
        plt.scatter(X_projected[y == 1], np.zeros_like(X_projected[y == 1]), label='Class 1', marker='x')
        plt.title('Projected Data')
        plt.legend()

        plt.tight_layout()
        plt.show()


# {% if trained_model.roc_curve_image %}
# <img src="{{ trained_model.roc_curve_image.url }}" alt="ROC Curve">
# {% else %}
# <p>No ROC curve image available</p>
# {% endif %}