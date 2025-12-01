
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def get_fences(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    return lower_fence, upper_fence


def calculate_metrics(y_true, y_pred):
    """
    Calcula métricas de classificação para um modelo.
    
    Parameters:
    -----------
    model_name : str
        Nome do modelo
    y_true : array-like
        Classes reais
    y_pred : array-like
        Classes previstas
        
    Returns:
    --------
    dict
        Dicionário com as métricas: nome do modelo, acurácia, precisão, 
        recall, f1-score, specificity e AUC
    """
    # Calcular matriz de confusão
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calcular métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    auc = roc_auc_score(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'auc': auc
    }