import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Union
import warnings


def accuracy_off1_macro(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> float:
    """Macro-averaged 1-off accuracy"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if labels is None:
        labels = np.unique(y_true)
    
    class_accuracies = []
    for label in labels:
        class_mask = y_true == label
        if np.sum(class_mask) > 0:  # Avoid empty classes
            class_pred = y_pred[class_mask]
            class_true = y_true[class_mask]
            # Calculate 1-off accuracy for this class
            abs_diff = np.abs(class_pred - class_true)
            class_acc = np.mean(abs_diff <= 1)
            class_accuracies.append(class_acc)
    
    return np.mean(class_accuracies)  # Macro average


class ScottsPiQuadratic:
    """
    Implementation of Scott's Pi with Quadratic Weights for ordinal classification.
    
    Scott's Pi is a chance-corrected agreement measure that's particularly suitable
    for ordinal data with imbalanced class distributions. The quadratic weighting
    scheme penalizes distant misclassifications more severely than close ones.
    
    Mathematical Formula:
    π = (Po - Pe) / (1 - Pe)
    
    Where:
    - Po = Observed weighted agreement = Σᵢⱼ wᵢⱼpᵢⱼ
    - Pe = Expected weighted agreement = Σᵢⱼ wᵢⱼpᵢpⱼ, pᵢ = (pᵢ. + p.ᵢ)/2
    - wᵢⱼ = Quadratic weights = 1 - (i-j)²/(R-1)²
    """
    
    def __init__(self):
        self.weights_ = None
        self.confusion_matrix_ = None
        self.categories_ = None
    
    def _create_quadratic_weights(self, n_categories: int) -> np.ndarray:
        """
        Create quadratic weight matrix for ordinal categories.
        
        Formula: wᵢⱼ = 1 - (i-j)²/(R-1)²
        
        Args:
            n_categories: Number of ordinal categories
            
        Returns:
            Weight matrix where diagonal = 1, off-diagonal decreases quadratically
        """
        if n_categories <= 1:
            return np.ones((n_categories, n_categories))
    
        # Create a grid of indices
        i, j = np.ogrid[:n_categories, :n_categories]
        distance_squared = (i - j) ** 2
        max_distance_squared = (n_categories - 1) ** 2
        
        return 1.0 - (distance_squared / max_distance_squared)
    
    def _calculate_cell_probabilities(self, cm: np.ndarray) -> np.ndarray:
        """Calculate cell probabilities from confusion matrix."""
        total = np.sum(cm)
        if total == 0:
            raise ValueError("Confusion matrix is empty")
        return cm / total
    
    def _calculate_marginal_probabilities(self, p_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate joint marginal probabilities for Scott's Pi.
        
        Scott's Pi uses averaged marginals: pᵢ = (pᵢ. + p.ᵢ)/2
        This differs from Cohen's Kappa which uses separate marginals.
        """
        row_marginals = np.sum(p_matrix, axis=1)  # pᵢ.
        col_marginals = np.sum(p_matrix, axis=0)  # p.ᵢ
        
        # Scott's Pi uses joint proportions (averaged marginals)
        joint_marginals = (row_marginals + col_marginals) / 2
        
        return joint_marginals
    
    def _calculate_observed_agreement(self, p_matrix: np.ndarray, weights: np.ndarray) -> float:
        """
        Calculate observed weighted agreement: Po = Σᵢⱼ wᵢⱼpᵢⱼ
        """
        return np.sum(weights * p_matrix)
    
    def _calculate_expected_agreement(self, joint_marginals: np.ndarray, weights: np.ndarray) -> float:
        """
        Calculate expected weighted agreement: Pe = Σᵢⱼ wᵢⱼpᵢpⱼ
        
        For Scott's Pi, we use joint marginals: pᵢ = (pᵢ. + p.ᵢ)/2
        """
        expected_matrix = np.outer(joint_marginals, joint_marginals)
        return np.sum(weights * expected_matrix)
    
    def fit_score(self, y_true: Union[list, np.ndarray], y_pred: Union[list, np.ndarray]) -> float:
        """
        Calculate Scott's Pi with Quadratic Weights.
        
        Args:
            y_true: True ordinal labels
            y_pred: Predicted ordinal labels
            
        Returns:
            Scott's Pi coefficient (-1 to 1, where 1 = perfect agreement)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        # Get unique categories and create mapping
        all_categories = np.unique(np.concatenate([y_true, y_pred]))
        self.categories_ = all_categories
        n_categories = len(all_categories)
        
        # Create category mapping for 0-indexed confusion matrix
        category_to_idx = {cat: idx for idx, cat in enumerate(all_categories)}
        y_true_idx = np.array([category_to_idx[cat] for cat in y_true])
        y_pred_idx = np.array([category_to_idx[cat] for cat in y_pred])
        
        # Create confusion matrix
        self.confusion_matrix_ = confusion_matrix(y_true_idx, y_pred_idx, 
                                                labels=range(n_categories))
        
        # Create quadratic weights
        self.weights_ = self._create_quadratic_weights(n_categories)
        
        # Calculate cell probabilities
        p_matrix = self._calculate_cell_probabilities(self.confusion_matrix_)
        
        # Calculate joint marginal probabilities (Scott's Pi approach)
        joint_marginals = self._calculate_marginal_probabilities(p_matrix)
        
        # Calculate observed and expected agreements
        po = self._calculate_observed_agreement(p_matrix, self.weights_)
        pe = self._calculate_expected_agreement(joint_marginals, self.weights_)
        
        # Calculate Scott's Pi
        if pe == 1.0:
            if po == 1.0:
                return 1.0
            else:
                warnings.warn("Expected agreement is 1.0 but observed is not. "
                            "This may indicate perfect chance agreement.")
                return 0.0
        
        scotts_pi = (po - pe) / (1.0 - pe)
        
        return scotts_pi
    
    def get_weight_matrix(self) -> np.ndarray:
        """Return the quadratic weight matrix used in the last calculation."""
        if self.weights_ is None:
            raise ValueError("Must call fit_score first")
        return self.weights_.copy()
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Return the confusion matrix from the last calculation."""
        if self.confusion_matrix_ is None:
            raise ValueError("Must call fit_score first")
        return self.confusion_matrix_.copy()