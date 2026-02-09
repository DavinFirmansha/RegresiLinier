# ===================== TAB 5: COMPARISON + DRIFT + COMPLEXITY =====================
with tab5:
    st.markdown("## \U0001f3c6 Model Comparison")
    if st.session_state.comparison_df is None or len(st.session_state.comparison_df)==0: st.warning("Train first.");st.stop()
    comp=st.session_state.comparison_df.copy();preds=st.session_state.all_predictions;y_te=st.session_state.y_test;cn=st.session_state.class_names
    st.markdown("### Leaderboard")
    show_c=["model_name","accuracy","f1","mcc","roc_auc","cv_mean","cv_std","tuning_cv_score","n_params","model_size_kb","train_time"]
    st.dataframe(comp[[c for c in show_c if c in comp.columns]].round(4),use_container_width=True)
    b=comp.iloc[0]
    st.markdown(f'<div class="success-box"><b>\U0001f3c6 {b["model_name"]}</b> Acc={b["accuracy"]:.4f} F1={b.get("f1",0):.4f}</div>',unsafe_allow_html=True)
    st.markdown("### Metric Bars")
    m_opts=[m for m in ["accuracy","f1","mcc","roc_auc","cv_mean"] if m in comp.columns]
    sel_m=st.multiselect("Metrics:",m_opts,default=m_opts[:4],key="cmp_m")
    for met in sel_m:
        sdf=comp[["model_name",met]].dropna().sort_values(met,ascending=False)
        colors=["#27AE60" if i==0 else "#3498DB" for i in range(len(sdf))]
        fig=go.Figure(go.Bar(x=sdf["model_name"],y=sdf[met],marker_color=colors,text=sdf[met].round(4),textposition="outside"))
        fig.update_layout(title=met.title(),height=400);st.plotly_chart(fig,use_container_width=True)
    st.markdown("### Radar")
    rm=[m for m in ["accuracy","precision","recall","f1","mcc","balanced_accuracy"] if m in comp.columns]
    if len(rm)>=3 and len(comp)>=2:
        topn=st.slider("Top N:",2,min(10,len(comp)),min(5,len(comp)))
        fig_rd=go.Figure();clrs=px.colors.qualitative.Set2
        for i in range(topn):
            r=comp.iloc[i];vals=[r.get(m,0) for m in rm]+[r.get(rm[0],0)]
            fig_rd.add_trace(go.Scatterpolar(r=vals,theta=rm+[rm[0]],fill="toself",name=r["model_name"],line=dict(color=clrs[i%len(clrs)])))
        fig_rd.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])),height=550);st.plotly_chart(fig_rd,use_container_width=True)
    if "n_params" in comp.columns and "train_time" in comp.columns:
        st.markdown("### Model Complexity")
        st.plotly_chart(px.scatter(comp,x="model_size_kb",y="accuracy",size="train_time",color="model_name",text="model_name",title="Accuracy vs Size (bubble=time)").update_traces(textposition="top center").update_layout(height=500,xaxis_title="Model Size (KB)",yaxis_title="Accuracy"),use_container_width=True)
    st.markdown("### Data Drift Detection")
    if st.button("Check Train vs Test Drift",key="drift_btn"):
        X_tr=st.session_state.X_train;X_te=st.session_state.X_test;fn=st.session_state.feature_names
        drift_data=[]
        for i,fname in enumerate(fn):
            if i>=X_tr.shape[1]: break
            stat,pv=ks_2samp(X_tr[:,i],X_te[:,i])
            drift_data.append({"Feature":fname,"KS Statistic":round(stat,4),"p-value":round(pv,4),"Drift":"Yes" if pv<0.05 else "No"})
        ddf=pd.DataFrame(drift_data).sort_values("KS Statistic",ascending=False)
        st.dataframe(ddf,use_container_width=True)
        nd=len(ddf[ddf["Drift"]=="Yes"])
        if nd>0: st.markdown(f'<div class="warn-box">\u26a0\ufe0f {nd} features show significant drift (p<0.05)</div>',unsafe_allow_html=True)
        else: st.markdown('<div class="success-box">\u2705 No significant drift detected</div>',unsafe_allow_html=True)
    st.markdown("### ROC Overlay")
    nc4=len(cn) if cn else 2;fig_ra=go.Figure();clrs4=px.colors.qualitative.Set2+px.colors.qualitative.Pastel
    for i,mn in enumerate(comp["model_name"].tolist()):
        yp2=preds.get(mn,{}).get("y_prob")
        if yp2 is not None:
            try:
                if nc4==2: fpr,tpr,_=roc_curve(y_te,yp2[:,1]);fig_ra.add_trace(go.Scatter(x=fpr,y=tpr,name=f"{mn} ({auc(fpr,tpr):.3f})",line=dict(color=clrs4[i%len(clrs4)])))
            except: pass
    fig_ra.add_trace(go.Scatter(x=[0,1],y=[0,1],line=dict(dash="dash",color="gray"),showlegend=False))
    fig_ra.update_layout(title="ROC All",height=500);st.plotly_chart(fig_ra,use_container_width=True)
    st.markdown("### Heatmap")
    hc=[c for c in ["accuracy","f1","mcc","roc_auc","cv_mean"] if c in comp.columns]
    if hc:
        hdf=comp.set_index("model_name")[hc].dropna(axis=1,how="all")
        st.plotly_chart(px.imshow(hdf,text_auto=".3f",color_continuous_scale="RdYlGn",aspect="auto").update_layout(height=max(300,len(comp)*40)),use_container_width=True)
    st.markdown("### Per-Fold CV Scores")
    cvf_model=st.selectbox("Model:",list(st.session_state.trained_models.keys()),key="cvf_m")
    if st.button("Show Folds",key="cvf_btn"):
        try:
            cv_scores=cross_val_score(st.session_state.trained_models[cvf_model],st.session_state.X_train,st.session_state.y_train,cv=5,scoring="accuracy")
            fold_df=pd.DataFrame({"Fold":[f"Fold {i+1}" for i in range(len(cv_scores))],"Accuracy":cv_scores})
            fig_cv=go.Figure(go.Bar(x=fold_df["Fold"],y=fold_df["Accuracy"],text=fold_df["Accuracy"].round(4),textposition="outside",marker_color=px.colors.qualitative.Set2[:len(cv_scores)]))
            fig_cv.add_hline(y=cv_scores.mean(),line_dash="dash",annotation_text=f"Mean={cv_scores.mean():.4f}")
            fig_cv.update_layout(height=400,title=f"Per-Fold: {cvf_model}");st.plotly_chart(fig_cv,use_container_width=True)
        except Exception as e: st.warning(str(e))

# ===================== TAB 6: ANALYSIS + SHAP =====================
with tab6:
    st.markdown("## \U0001f50d Deep Analysis")
    if not st.session_state.trained_models: st.warning("Train first.");st.stop()
    trained=st.session_state.trained_models;preds=st.session_state.all_predictions
    X_tr=st.session_state.X_train;X_te=st.session_state.X_test;y_tr=st.session_state.y_train;y_te=st.session_state.y_test
    cn=st.session_state.class_names;fn=st.session_state.feature_names;mnames=list(trained.keys())
    st.markdown("### Learning Curves")
    lc_model=st.selectbox("Model:",mnames,key="lc_m")
    if st.button("Plot",key="lc_btn"):
        with st.spinner("..."):
            try:
                tsz,tr_sc,te_sc=learning_curve(trained[lc_model],X_tr,y_tr,cv=5,n_jobs=-1,train_sizes=np.linspace(0.1,1.0,10),scoring="accuracy",random_state=42)
                fig_lc=go.Figure()
                fig_lc.add_trace(go.Scatter(x=tsz,y=tr_sc.mean(axis=1),mode="lines+markers",name="Train"))
                fig_lc.add_trace(go.Scatter(x=tsz,y=te_sc.mean(axis=1),mode="lines+markers",name="Val"))
                fig_lc.update_layout(title=f"Learning Curve: {lc_model}",height=450);st.plotly_chart(fig_lc,use_container_width=True)
            except Exception as e: st.warning(str(e))
    st.markdown("### Calibration (Binary)")
    nc5=len(np.unique(y_te))
    if nc5==2 and st.button("Plot Calibration",key="cal_btn"):
        fig_cal=go.Figure();clrs5=px.colors.qualitative.Set2
        for i,mn in enumerate(mnames):
            yp3=preds.get(mn,{}).get("y_prob")
            if yp3 is not None:
                try: fp,mp2=calibration_curve(y_te,yp3[:,1],n_bins=10);fig_cal.add_trace(go.Scatter(x=mp2,y=fp,mode="lines+markers",name=mn,line=dict(color=clrs5[i%len(clrs5)])))
                except: pass
        fig_cal.add_trace(go.Scatter(x=[0,1],y=[0,1],line=dict(dash="dash",color="gray"),name="Perfect"))
        fig_cal.update_layout(title="Calibration",height=500);st.plotly_chart(fig_cal,use_container_width=True)
    st.markdown("### SHAP Explainability")
    if HAS_SHAP:
        shap_model=st.selectbox("Model:",mnames,key="shap_m")
        if st.button("Compute SHAP",key="shap_btn"):
            with st.spinner("Computing SHAP values (may take a moment)..."):
                try:
                    mdls=trained[shap_model]
                    X_sample=X_te[:min(200,len(X_te))]
                    if hasattr(mdls,"predict_proba"):
                        try: explainer=shap.TreeExplainer(mdls)
                        except: explainer=shap.KernelExplainer(mdls.predict_proba,shap.kmeans(X_tr,min(50,len(X_tr))))
                    else: explainer=shap.KernelExplainer(mdls.predict,shap.kmeans(X_tr,min(50,len(X_tr))))
                    shap_values=explainer.shap_values(X_sample)
                    if isinstance(shap_values,list): sv=np.abs(shap_values[0]) if len(shap_values)>0 else np.abs(shap_values)
                    elif shap_values.ndim==3: sv=np.abs(shap_values).mean(axis=2)
                    else: sv=np.abs(shap_values)
                    shap_imp=sv.mean(axis=0)
                    if len(shap_imp)==len(fn):
                        shap_df=pd.DataFrame({"Feature":fn,"SHAP":shap_imp}).sort_values("SHAP",ascending=True).tail(20)
                        st.plotly_chart(px.bar(shap_df,y="Feature",x="SHAP",orientation="h",color="SHAP",color_continuous_scale="Reds",title=f"SHAP: {shap_model}").update_layout(height=500),use_container_width=True)
                    st.success("SHAP computed!")
                except Exception as e: st.warning(f"SHAP error: {e}")
    else: st.info("pip install shap for SHAP explainability")
    st.markdown("### Cross-Val Predictions")
    cvp_model=st.selectbox("Model:",mnames,key="cvp_m")
    if st.button("CV Predict",key="cvp_btn"):
        with st.spinner("..."):
            try:
                cvp=cross_val_predict(trained[cvp_model],X_tr,y_tr,cv=5)
                cmcv=confusion_matrix(y_tr,cvp);clbl3=cn if len(cn)==cmcv.shape[0] else [str(i) for i in range(cmcv.shape[0])]
                co2=st.columns(2)
                with co2[0]: st.plotly_chart(px.imshow(cmcv,text_auto=True,color_continuous_scale="Blues",x=clbl3,y=clbl3).update_layout(title=f"CV: {cvp_model}",height=400),use_container_width=True)
                with co2[1]: st.dataframe(pd.DataFrame(classification_report(y_tr,cvp,target_names=clbl3,output_dict=True)).T.round(4),use_container_width=True)
            except Exception as e: st.warning(str(e))
    st.markdown("### Statistical Significance")
    if len(mnames)>=2:
        sig_m1=st.selectbox("Model A:",mnames,index=0,key="sig1")
        sig_m2=st.selectbox("Model B:",mnames,index=min(1,len(mnames)-1),key="sig2")
        if st.button("Test",key="sig_btn"):
            p1=preds.get(sig_m1,{}).get("y_pred");p2=preds.get(sig_m2,{}).get("y_pred")
            if p1 is not None and p2 is not None:
                c1_arr=(p1==y_te).astype(int);c2_arr=(p2==y_te).astype(int)
                diff=c1_arr-c2_arr;n_diff=np.sum(diff!=0)
                if n_diff>0:
                    n_plus=np.sum(diff>0)
                    bt=binomtest(n_plus,n_diff,0.5)
                    st.markdown(f"**McNemar-like test:** {sig_m1} correct where {sig_m2} wrong = {n_plus}, vice versa = {n_diff-n_plus}")
                    st.markdown(f"**p-value = {bt.pvalue:.6f}** {'(Significant)' if bt.pvalue<0.05 else '(Not significant)'}")
                else: st.info("Models have identical predictions.")

# ===================== TAB 7: PREDICT =====================
with tab7:
    st.markdown("## \U0001f52e Prediction")
    if not st.session_state.trained_models: st.warning("Train first.");st.stop()
    trained=st.session_state.trained_models;fn=st.session_state.feature_names;cn=st.session_state.class_names
    pred_model=st.selectbox("Model:",list(trained.keys()),key="pred_m")
    st.markdown("### Input Features")
    pred_method=st.radio("Input:",["Manual","Upload CSV"],horizontal=True)
    if pred_method=="Manual":
        vals={}
        ncols_p=min(4,len(fn))
        cols_p=st.columns(ncols_p)
        for i,f in enumerate(fn): vals[f]=cols_p[i%ncols_p].number_input(f,value=0.0,format="%.4f",key=f"pred_{f}")
        if st.button("Predict",key="pred_btn",type="primary"):
            X_new=np.array([[vals[f] for f in fn]])
            if st.session_state.scaler_obj: X_new=st.session_state.scaler_obj.transform(X_new)
            mdl=trained[pred_model]
            pred=mdl.predict(X_new)[0]
            label=cn[int(pred)] if int(pred)<len(cn) else str(pred)
            st.markdown(f'<div class="success-box"><h2>\U0001f3af Prediction: <b>{label}</b></h2></div>',unsafe_allow_html=True)
            if hasattr(mdl,"predict_proba"):
                try:
                    proba=mdl.predict_proba(X_new)[0]
                    prob_df=pd.DataFrame({"Class":cn[:len(proba)],"Probability":proba}).sort_values("Probability",ascending=False)
                    st.plotly_chart(px.bar(prob_df,x="Class",y="Probability",color="Probability",color_continuous_scale="Viridis",title="Class Probabilities").update_layout(height=350),use_container_width=True)
                except: pass
    else:
        up_pred=st.file_uploader("Upload:",type=["csv"],key="pred_upload")
        if up_pred and st.button("Predict All",key="pred_all_btn",type="primary"):
            try:
                pdf_new=pd.read_csv(up_pred)
                X_batch=pdf_new[fn].values.astype(float) if all(f in pdf_new.columns for f in fn) else pdf_new.values[:,:len(fn)].astype(float)
                if st.session_state.scaler_obj: X_batch=st.session_state.scaler_obj.transform(X_batch)
                mdl=trained[pred_model]
                preds_batch=mdl.predict(X_batch)
                labels=[cn[int(p)] if int(p)<len(cn) else str(p) for p in preds_batch]
                pdf_new["Prediction"]=labels
                if hasattr(mdl,"predict_proba"):
                    try:
                        proba_batch=mdl.predict_proba(X_batch)
                        for ci,cname in enumerate(cn[:proba_batch.shape[1]]): pdf_new[f"P({cname})"]=proba_batch[:,ci].round(4)
                    except: pass
                st.dataframe(pdf_new,use_container_width=True)
                csv_out=pdf_new.to_csv(index=False)
                st.download_button("Download Results",csv_out,"predictions.csv","text/csv")
            except Exception as e: st.error(str(e))

# ===================== TAB 8: EXPORT =====================
with tab8:
    st.markdown("## \U0001f4be Export & Pipeline")
    if not st.session_state.trained_models: st.warning("Train first.");st.stop()
    trained=st.session_state.trained_models;preds=st.session_state.all_predictions
    st.markdown("### Download Model")
    exp_model=st.selectbox("Model:",list(trained.keys()),key="exp_m")
    if st.button("Prepare Download",key="exp_btn"):
        mdl=trained[exp_model]
        buf=io.BytesIO();pickle.dump(mdl,buf);buf.seek(0)
        st.download_button(f"Download {exp_model} (.pkl)",buf,f"{exp_model.replace(' ','_')}.pkl","application/octet-stream")
    st.markdown("### Download All Results")
    if st.session_state.comparison_df is not None and len(st.session_state.comparison_df)>0:
        csv_comp=st.session_state.comparison_df.to_csv(index=False)
        st.download_button("Download Comparison CSV",csv_comp,"model_comparison.csv","text/csv")
    st.markdown("### Generate Pipeline Code")
    code_model=st.selectbox("Model:",list(trained.keys()),key="code_m")
    if st.button("Generate",key="code_btn"):
        bp=preds.get(code_model,{}).get("best_params",{})
        pp=st.session_state.get("pp_info",{})
        code=generate_pipeline_code(pp,code_model,bp if isinstance(bp,dict) else {})
        st.code(code,language="python")
        st.download_button("Download .py",code,f"pipeline_{code_model.replace(' ','_')}.py","text/plain")
    st.markdown("### Session Summary")
    if st.session_state.comparison_df is not None:
        st.json({
            "n_models":len(st.session_state.trained_models),
            "n_features":len(st.session_state.feature_names),
            "n_train":len(st.session_state.X_train) if st.session_state.X_train is not None else 0,
            "n_test":len(st.session_state.X_test) if st.session_state.X_test is not None else 0,
            "n_classes":len(st.session_state.class_names),
            "split_mode":st.session_state.split_mode,
            "best_model":st.session_state.comparison_df.iloc[0]["model_name"] if len(st.session_state.comparison_df)>0 else "N/A",
            "best_accuracy":round(st.session_state.comparison_df.iloc[0]["accuracy"],4) if len(st.session_state.comparison_df)>0 else 0,
            "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
