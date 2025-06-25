import json
import plotly.graph_objects as go
import plotly.io as pio

log_actor = "./split.txt"
time_max = 155000

times = []
scores = [[], [], [], []]
ranks = [[], [], [], []]
offset = 0

with open(log_actor, mode="r") as f:
    line = f.readline().strip().replace("\'", "\"")
    while line:
        print(line)
        json_dict = json.loads(line)
        if "plus" in json_dict:
            offset += json_dict["plus"]
            line = f.readline().strip().replace("\'", "\"")
            continue
        if json_dict["time"] > time_max:
            break
        times.append((json_dict["time"] + offset) / 60)
        scores[0].append(json_dict["scores"][0])
        scores[1].append(json_dict["scores"][1])
        scores[2].append(json_dict["scores"][2])
        scores[3].append(json_dict["scores"][3])
        ranks[0].append(json_dict["ranks"][0])
        ranks[1].append(json_dict["ranks"][1])
        ranks[2].append(json_dict["ranks"][2])
        ranks[3].append(json_dict["ranks"][3])
        line = f.readline().strip().replace("\'", "\"")

print(f"last time: {times[-1]}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=times, y=scores[3], mode="lines", name="score(Shanten)", yaxis="y1"))
fig.add_trace(go.Scatter(x=times, y=ranks[3], mode="lines", name="rank(Shanten)", yaxis="y2"))

fig.add_shape(type="line", x0=0, x1=max(times), y0=25000, y1=25000)

fig.update_layout(
    xaxis_title='time[min]',
    yaxis1=dict(side="left", range=(0.0, 50000.0)),
    yaxis2=dict(side="right", range=(0.0, 5.0), showgrid=False, overlaying="y"),
    yaxis1_title="final score (100 games Avg)",
    yaxis2_title="final rank (100 games Avg)",
    margin=dict(l=30, r=30, t=30, b=30),
)

fig.update_yaxes(tickformat=',')

pio.write_image(fig, "plot.png", width=1080, height=720)