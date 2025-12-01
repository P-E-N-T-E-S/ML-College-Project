FROM databricks/trendz-analytics

WORKDIR /mlflow

RUN pip install boto3 scikit-learn

EXPOSE 8080

CMD ["./start_mlflow_ui.sh"]
