#DEEP RNN
set_seeds()
model = create_deep_rnn_model(
            hl=2, hu=50, layer='SimpleRNN',
            features=len(data.columns),
            dropout=True, rate=0.3)

model.summary()

model.fit(g, epochs=200, steps_per_epoch=10,
          verbose=False, class_weight=cw(train_y))

y = np.where(model.predict(g_, batch_size=None) > 0.5,
             1, 0).flatten()


np.bincount(y)

accuracy_score(test_y[lags:], y)