def dropdown_by_category(top_categories):
    """Generates the structure needed for a Plotly dropdown"""
    return list([dict(
        type='dropdown',
        active=0,
        y=1.15,
        x=1.02,
        buttons=list([
            dict(
                label=top_categories.values[0],
                method='update',
                args=[
                    {'visible': [True, False, False]}, 
                    {'title': f'Top words by "{top_categories.values[0]}" category'}
                ]
            ),
            dict(
                label=top_categories.values[1],
                method='update',
                args=[
                    {'visible': [False, True, False]}, 
                    {'title': f'Top words by "{top_categories.values[1]}" category'}
                ]
            ),
            dict(
                label=top_categories.values[2],
                method='update',
                args=[
                    {'visible': [False, False, True]}, 
                    {'title': f'Top words by "{top_categories.values[2]}" category'}
                ]
            ),            
        ]),
    )])    