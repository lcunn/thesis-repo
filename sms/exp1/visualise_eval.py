import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt  # Only needed if you're using other matplotlib features

def visualize_avg_recall(results, k_list=[1, 3, 5, 10, 25, 50, 100]):
    """
    Create a line plot to visualize the average recall for different augmentation types using Plotly.
    
    Args:
        results: The output dictionary from evaluate_top_k function for a single model
        k_list: List of k values used in the evaluation
    """
    aug_types = list(results.keys())
    
    # Generate distinct colors for each augmentation type
    colors = px.colors.qualitative.Plotly[:len(aug_types)]
    
    fig = go.Figure()
    
    all_avg_recalls = []
    
    for aug_type, color in zip(aug_types, colors):
        avg_recalls = [results[aug_type][k]['avg_recall'] for k in k_list]
        fig.add_trace(go.Scatter(
            x=k_list,
            y=avg_recalls,
            mode='lines+markers',
            name=aug_type,
            line=dict(color=color, width=2),
            marker=dict(size=8)
        ))
        all_avg_recalls.append(avg_recalls)
    
    # Calculate and plot the average of all augmentations
    avg_of_avgs = np.mean(all_avg_recalls, axis=0)
    fig.add_trace(go.Scatter(
        x=k_list,
        y=avg_of_avgs,
        mode='lines+markers',
        name='Average of all',
        line=dict(color='black', width=2, dash='dash'),
        marker=dict(symbol='square', size=10)
    ))
    
    # Update layout for aesthetics
    fig.update_layout(
        title='Average Recall vs k for Different Augmentation Types',
        xaxis_title='k',
        yaxis_title='Average Recall',
        xaxis_type='log',
        xaxis=dict(
            tickmode='array',
            tickvals=k_list,
            ticktext=[str(k) for k in k_list]
        ),
        legend=dict(
            title='Augmentation Type',
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        ),
        hovermode='x unified',
        template='plotly_white',
        width=800,
        height=600
    )
    
    # Save the plot as a static image file
    fig.write_image("avg_recall_results.png", scale=2)