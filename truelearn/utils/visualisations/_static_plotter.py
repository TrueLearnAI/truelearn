import plotly.graph_objects as go
import numpy as np

#Bar Graphs
x_values = ['1027: Animal Communication', '6861: Data-Driven Programming', '4212: Chess', 
'10270: Football', '420: Adolescence']
y_values = [0.57,0.84,0.23,0.15,0.09]
variance = 'rgba(0.10,0.02,0.40,0.52,0.63)'
fig = go.Figure(go.Bar(x=x_values, y=y_values, marker=dict(color=variance)))
fig.update_layout(title='Skill Level in different Subjects', xaxis_title='Subjects', yaxis_title='TrueSkill rating')
fig.show()

# Line Graphs
x_values = ['Jan 2023', 'Feb 2023', 'Mar 2023', 'Apr 2023', 'May 2023']
y_values = [0.09,0.15,0.23,0.57,0.84]
variance = np.random.rand(10)
line_trace = go.Scatter(
    x=x_values,
    y=y_values,
    mode='lines+markers',
    marker=dict(
        size=5,
        color=variance,
        colorscale='rgba',
        showscale=True
    ),
    line=dict(width=2)
)
layout = go.Layout(
    title='Skill Level over time',
    xaxis=dict(title='Time'),
    yaxis=dict(title='TrueSkill rating')
)
fig = go.Figure(data=line_trace, layout=layout)
fig.show()

# Bubble Charts
fig = go.Figure(data=[go.Scatter(
        x= ['1027: Animal Communication', '6861: Data-Driven Programming', '4212: Chess', 
        '10270: Football', '420: Adolescence'], y= [0.57,0.84,0.23,0.15,0.09],
        mode='variance',
        variance=[0.10,0.02,0.40,0.52,0.63])
])

fig.show()

# Pie Charts
labels = ['1027: Animal Communication', '6861: Data-Driven Programming', '4212: Chess', 
'10270: Football', '420: Adolescence']
values = [0.57,0.84,0.23,0.15,0.09]

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.show()

# Bullet Charts
fig = go.Figure(go.Indicator(
    mode = "number+gauge+delta", value = 220,
    domain = {'x': [0, 1], 'y': [0, 1]},
    delta = {'reference': 280, 'position': "top"},
    title = {'text':"<b>TrueSkill_rating</b><br><span style='color: gray; font-size:0.8em'>0-1</span>", 'font': {"size": 14}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 300]},
        'threshold': {
            'line': {'color': "red", 'width': 2},
            'thickness': 0.75, 'value': 270},
        'bgcolor': "white",
        'steps': [
            {'range': [0, 150], 'color': "cyan"},
            {'range': [150, 250], 'color': "royalblue"}],
        'bar': {'color': "darkblue"}}))
fig.update_layout(height = 250)
fig.show()