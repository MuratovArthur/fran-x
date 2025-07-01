import collections


class Evaluator:
    Entity = collections.namedtuple("Entity", ["text", "type", "start"])

    def __init__(self, y_true, y_pred):
        """
        y_true: list[list[Entity]]
            For each text in the dataset, a list of ground-truth entities.
        y_pred: list[list[Entity]]
            For each text in the dataset, a list of predicted entities.

        Example
        -------
        y_true = [[Entity('a', 'b', 0), Entity('b', 'b', 2)]]
        y_pred = [[Entity('b', 'b', 2)]]
        """
        self.y_true = y_true
        self.y_pred = y_pred
        if len(self.y_true) != len(self.y_pred):
            raise ValueError(
                "Bad input shape: y_true and y_pred should have the same length."
            )

    @staticmethod
    def has_overlap(x, y):
        """
        x: Entity
            First entity to compare
        y: Entity
            Second entity to compare

        Determines whether the text of two entities overlap according their starting positions and length of the text.
         This function is symmetric
         self.has_overlap(x, y) == self.has_overlap(y, x)

        :return
        is_correct_type: bool
            True if x and y have are overlapped in the text. Otherwise, False.
        """
        # YOUR CODE HERE #
        if x is None or y is None:
            return False
            
        x_end = x.start + len(x.text)
        y_end = y.start + len(y.text)
        
        # Check if one range is completely before or after the other
        if x_end <= y.start or y_end <= x.start:
            return False
        
        return True


    @staticmethod
    def correct_text(x, y):
        """
        x: Entity
            First entity to compare
        y: Entity
            Second entity to compare

        Assert that text and starting position of two entities are the same.

        :return
        is_correct_text: bool
            True if x and y have the same text and same start. Otherwise, False.
        """
        # YOUR CODE HERE #
        if x is None or y is None:
            return False
            
        return x.text == y.text and x.start == y.start


    @staticmethod
    def correct_type(x, y):
        """
        x: Entity
            First entity to compare
        y: Entity
            Second entity to compare

        Assert entity types match and that there is an overlap in the text of the two entities.

        :return
        is_correct_type: bool
            True if type of x and y are equal, and they have overlap. Otherwise, False.
        """
        # YOUR CODE HERE #
        if x is None or y is None:
            return False
            
        return x.type == y.type and Evaluator.has_overlap(x, y)

    @staticmethod
    def count_correct(gt_entities, pred_entities):
        """
        gt_entities
        pred_entities

        Computes the count of correctly predicted entities on two axes: type and text.

        Parameters
        ----------
        gt_entities: list of Entity
            The list of ground truth entities.
        pred_entities: list of Entity
            The list of predicted entities.

        :return
        count_text: int
            The number of entities predicted where the text matches exactly.
        count_type: int
            The number of entities where the type is correctly predicted and the text overlaps.
        """
        # YOUR CODE HERE #
        if gt_entities is None or pred_entities is None:
            return (0, 0)
            
        count_text = 0
        count_type = 0
        
        for gt_entity in gt_entities:
            for pred_entity in pred_entities:
                if Evaluator.correct_text(gt_entity, pred_entity):
                    count_text += 1
                if Evaluator.correct_type(gt_entity, pred_entity):
                    count_type += 1
                    
        return (count_text, count_type)


    @staticmethod
    def precision(correct, actual):
        if actual == 0:
            return 0
        return correct / actual

    @staticmethod
    def recall(correct, possible):
        if possible == 0:
            return 0
        return correct / possible

    @staticmethod
    def f1(p, r):
        if p + r == 0:
            return 0
        return 2 * (p * r) / (p + r)

    def evaluate(self):
        """
        Evaluate classification results for a whole dataset. Each row corresponds to one text in the
        dataset.

        :return
        f1: float
            Micro-averaged F1 score
        precision: float
            Micro-averaged precision
        recall: float
            Micro-averaged recall
        """
        correct, actual, possible = 0, 0, 0

        for x, y in zip(self.y_true, self.y_pred):
            correct += sum(Evaluator.count_correct(x, y))
            # multiply by two to account for both type and text
            possible += len(x) * 2
            actual += len(y) * 2
        precision = Evaluator.precision(correct, actual)
        recall = Evaluator.recall(correct, possible)
        f1 = Evaluator.f1(precision, recall)
        return f1, precision, recall