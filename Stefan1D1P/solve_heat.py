from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import os

from Stefan1D1P.Stefan_models_tf import Sampler
from Stefan1D1P.heat_eq import Heat

ALPHA = 0.1

def run_heat():

    # Analytical solution
    def u(z):
        x = z[:, 0:1]
        t = z[:, 1:2]
        u = np.sin(np.pi*x)*np.exp(-(ALPHA**2)*(np.pi**2)*t)
        return u

    ic_coords =  np.array([[0.0, 0.0],
                        [1.0, 0.0]])
    bcl_coords = np.array([[0.0, 0.0],
                        [0.0, 1.0]])
    bcr_coords = np.array([[1.0, 0.0],
                        [1.0, 1.0]])
    dom_coords = np.array([[0.0, 0.0],
                        [1.0, 1.0]])

    layers = [2, 100, 100, 100, 1]
    ics_sampler = Sampler(2, ic_coords, func=lambda X: np.sin(np.pi*X[:, 0:1]), name="Initial Condition")
    bcls_sampler = Sampler(2, bcl_coords, lambda x: 0, name="Boundary Condition Left")
    bcrs_sampler = Sampler(2, bcr_coords, lambda x: 0, name="Boundary Condition Right")
    res_sampler = Sampler(2, dom_coords, func=u, name="Forcing")

    # X, y = ics_sampler.sample(1000)
    # x = X[:, 0:1]
    # plt.scatter(x, y)
    # plt.show()

    model = Heat(layers, ics_sampler, bcls_sampler, bcrs_sampler, res_sampler, ALPHA)

    # weights path
    relative_path = os.path.join('results', 'heat_eq')
    current_directory = os.getcwd()
    weights_dir = os.path.join(current_directory, relative_path)
    weights_file = os.path.join(weights_dir, 'model_weights.ckpt')

    # Load model
    if os.path.exists(weights_file):
        model.load_weights(weights_file)

    model.train(n_iter=40000, batch_size=128)

    ### Save Model ###
    ####################
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        
    model.save_weights(weights_file)
    print(f"Model weights saved to: {weights_file}")

    # Test data
    nn = 200
    x = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
    t = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    # Exact solutions
    u_star = u(X_star)

    # Predictions
    u_pred = model.predict(X_star)

    # Errors
    error = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print('Relative L2 error: {:.2e}'.format(error))
        
    U_star = griddata(X_star, u_star.flatten(), (X, T), method='cubic')
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    vmin = min(U_star.min(), U_pred.min())
    vmax = max(U_star.max(), U_pred.max())
                
    fig_1 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(X, T, U_star, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.title('Exact $u(x,t)$')

    plt.subplot(1, 3, 2)
    plt.pcolor(X, T, U_pred, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.title('Predicted $u(x,t)$')

    plt.subplot(1, 3, 3)
    plt.pcolor(X, T, np.abs(U_star - U_pred), cmap='jet')
    plt.colorbar(format='%.0e')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.title('Absolute Error')

    plt.tight_layout()
    plt.show()

    plt.plot(model.loss_res_log, label="Residual")
    plt.plot(model.loss_ics_log, label="Initial")
    plt.plot(model.loss_bcls_log, label="Left Boundary")
    plt.plot(model.loss_bcrs_log, label="Right Boundary")
    plt.title('Loss over time')
    plt.legend()
    plt.show()