-- get metrics from run
SELECT step, value AS val_loss FROM metrics WHERE metrics.key='val_loss' AND
metrics.run_uuid='0b09d59f2f4d48fda90f578bdfe2c855' ORDER BY step LIMIT 5;



-- get multiple metrics from run

-- self joins needed since all metrics are stored in 
SELECT t.step, 
round( t.value::numeric, 3) AS train_loss, 
ROUND ( v.value::numeric, 3) AS valid_loss,
ROUND ( ta.value::numeric, 3) AS train_acc,
ROUND ( va.value::numeric, 3) AS val_acc
FROM metrics AS t, metrics AS v, metrics as ta, metrics as va
WHERE t.key='train_loss' AND t.run_uuid='0b09d59f2f4d48fda90f578bdfe2c855'
AND v.key='val_loss' AND v.run_uuid=t.run_uuid
AND ta.key='train_acc' AND ta.run_uuid=t.run_uuid
AND va.key='val_acc' AND va.run_uuid=t.run_uuid
AND t.step = v.step
AND t.step = ta.step
AND t.step = va.step
AND t.step % 250 = 0
ORDER BY t.step
LIMIT 100;

-- links from child to parent runs 
select key, value, run_uuid from tags
where key = 'mlflow.parentRunId';

-- get numeric parameters
