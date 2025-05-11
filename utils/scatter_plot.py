from matplotlib.ticker import FuncFormatter

def scatter_prediction(y_test, y_pred, model_name):
    import matplotlib.pyplot as plt
    import numpy as np

    # Scatter plot of actual vs predicted values
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  
    plt.xlabel('Actual Price (€)')
    plt.ylabel('Predicted Price (€)')
    plt.title(f'Predicted vs. Actual Car Prices - {model_name}')

    # Format the x and y axes to show euros
    def euros(x, pos):
        if x >= 1000000:
            return f'€{x/1000000:.1f}M'
        elif x >= 1000:
            return f'€{x/1000:.0f}K'
        else:
            return f'€{int(x)}'
    
    plt.gca().xaxis.set_major_formatter(FuncFormatter(euros))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(euros))

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.show()
