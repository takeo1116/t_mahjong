import json
import plotly.graph_objects as go
import plotly.io as pio

log_actor = "./actor_log.txt"
time_max = 155000

times = []
scores = [[], [], [], []]
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
        line = f.readline().strip().replace("\'", "\"")

print(f"last time: {times[-1]}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=times, y=scores[0], mode="lines", name="Learn0"))
fig.add_trace(go.Scatter(x=times, y=scores[1], mode="lines", name="Learn1"))
fig.add_trace(go.Scatter(x=times, y=scores[2], mode="lines", name="Learn2"))
fig.add_trace(go.Scatter(x=times, y=scores[3], mode="lines", name="Shanten"))

fig.add_shape(type="line", x0=0, x1=max(times), y0=25000, y1=25000)

fig.update_layout(
    xaxis_title='time[min]',
    yaxis_title='final score (100 games average)',
    margin=dict(l=30, r=30, t=30, b=30),
)

fig.update_yaxes(tickformat=',')

pio.write_image(fig, "plot.png", width=1080, height=720)